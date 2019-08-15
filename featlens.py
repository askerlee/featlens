import torch
import torch.nn as nn
import torch.nn.functional as F
from dotmap import DotMap
import os
import pdb
import time
import cv2
import numpy as np
import torchvision.models as models
import inspect

from PIL import Image
import math
import re
import socket
hostname = socket.gethostname()
import sys

np.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)
torch.set_printoptions(sci_mode=False)

if hostname == 'SMG05-DJW':
    os.environ['IMAGENET'] = '/home/shaohua/hebbian' # '/data/home/shaohua/ILSVRC2012'
elif hostname == 'toastbox':
    os.environ['IMAGENET'] = '/ssd/ILSVRC2012'
elif hostname == 'm10':
    os.environ['IMAGENET'] = '/data/home/shaohua'
elif hostname == 'workstation' or hostname == 'workstation2':
    os.environ['IMAGENET'] = '/data/shaohua'
else:
    print("Unknown hostname '%s'. Please specify 'IMAGENET' manually." %hostname)
    exit(0)


def pearson(t1, t2, dim=-1):
    if dim == -1:
        t1flat = t1.view(-1)
        t2flat = t2.view(-1)
        t1flatz = t1flat - t1flat.mean()
        t2flatz = t2flat - t2flat.mean()
        norm1 = (t1flatz**2).sum().sqrt()
        norm2 = (t2flatz**2).sum().sqrt()
        norm1[norm1 < 1e-5] = 1
        norm2[norm2 < 1e-5] = 1
        
        corr = (t1flatz * t2flatz).sum() / (norm1 * norm2)
        return corr.item()
        
    elif dim == 0:
        t1flat = t1.view(t1.shape[0], -1)
        t2flat = t2.view(t2.shape[0], -1)
        t1flatz = t1flat - t1flat.mean(dim=1, keepdim=True)
        t2flatz = t2flat - t2flat.mean(dim=1, keepdim=True)
        norm1 = torch.pow(t1flatz, 2).sum(dim=1).sqrt()
        norm2 = torch.pow(t2flatz, 2).sum(dim=1).sqrt()
        norm1[norm1 < 1e-5] = 1
        norm2[norm2 < 1e-5] = 1
        
        corr = (t1flatz * t2flatz).sum(dim=1) / (norm1 * norm2)
        return corr.data.cpu().numpy()
    else:
        raise NotImplementedError

# if dim=None, can be used on a pytorch tensor as well as a numpy array
def calc_tensor_stats(arr, do_print=False, arr_name=None, dim=None):
    if dim is None:
        arr_min = arr.min()
        arr_max = arr.max()
        arr_mean = arr.mean()
        arr_std  = arr.std()
        stats = arr_max.item(), arr.argmax(), arr_min.item(), arr.argmin(), arr_mean.item(), arr_std.item()
    else:
        arr_min = arr.min(dim=dim)[0]
        arr_max = arr.max(dim=dim)[0]
        arr_mean = arr.mean(dim=dim)
        arr_std  = arr.std(dim=dim)
        stats = arr_max, arr.argmax(), arr_min, arr.argmin(), arr_mean, arr_std
    
    if do_print:
        print("%s: max %.3f (%d) min %.3f (%d) mean %.3f std %.3f" %((arr_name,) + stats) )
    return stats

def rotate_tensor4d(x, angle):
    if angle == 0 or angle == 360:
        rot_x = x
    elif angle == 90:
        rot_x = x.transpose(2, 3).flip(2)
    elif angle == 180:
        rot_x = x.flip(2).flip(3)
    elif angle == 270:
        rot_x = x.transpose(2, 3).flip(3)
    else:
        raise NotImplementedError
    
    return rot_x
    
# x: feature maps of a batch (b,c,h,w)
# pos_b: indices to the batch dim. No rotation in this dim
# pos_hw: indices to the (h, w) dims. Need rotation in this plane
# w: x, h: y
# refer: https://discuss.pytorch.org/t/how-to-rotate-90-and-270-degrees-of-5d-tensor/22476/8
def value_at_rotated_coords(x, orig_b, orig_c, orig_h, orig_w, orig_HW, angle):
    orig_H, orig_W = orig_HW
    H, W = x.shape[2:]
    if angle == 0:
        rot_w = orig_w
        rot_h = orig_h
    elif angle == 90:
        rot_w = orig_h
        rot_h = H - orig_w - 1
    elif angle == 180:
        rot_w = W - orig_w - 1
        rot_h = H - orig_h - 1
    elif angle == 270:
        rot_w = W - orig_h - 1
        rot_h = orig_w
    else:
        raise NotImplementedError
    
    # rotate back to the original position (so that original coords apply), so 360 - angle
    invrot_x = rotate_tensor4d(x, 360 - angle)
    
    if orig_c is None:
        # sel_x1 = x[orig_b, :, rot_h, rot_w]
        sel_x2 = invrot_x[orig_b, :, orig_h, orig_w]
        return sel_x2
    else:
        # sel_x1 = x[orig_b, orig_c, rot_h, rot_w]
        sel_x2 = invrot_x[orig_b, orig_c, orig_h, orig_w]
        return sel_x2

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
        
class MixConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 dilation, groups, has_bias, num_comps=5, use_softmax=True, 
                 context_size=0, add_residual=False):
        super(MixConv, self).__init__()
        self.mix_convs = nn.ModuleList()
        self.num_comps = num_comps
        self.use_softmax = use_softmax and self.num_comps > 1
        self.in_channels, self.out_channels, self.orig_kernel_size, self.stride, self.orig_padding, \
            self.dilation, self.groups, self.has_bias = \
                in_channels, out_channels, kernel_size, stride, padding, \
                 dilation, groups, has_bias
                 
        self.padding = (self.orig_padding[0] + context_size, self.orig_padding[1] + context_size)
        self.kernel_size = (self.orig_kernel_size[0] + 2 * context_size, 
                            self.orig_kernel_size[1] + 2 * context_size)
        
        self.add_residual = add_residual
        
        if self.add_residual:
            C = 2
        else:
            C = 1
            
        for i in range(self.num_comps):
            self.mix_convs.append(nn.Conv2d(self.in_channels, self.out_channels * C, 
                                                self.kernel_size, self.stride, self.padding, 
                                                self.dilation, self.groups, bias=True)) # always use bias
            torch.nn.init.xavier_uniform_(self.mix_convs[-1].weight)
        
    def forward(self, x):
        feats = []
        for i in range(self.num_comps):
            feats.append(self.mix_convs[i](x))
        feats = torch.stack(feats, dim=0)
            
        if self.use_softmax:
            weights = torch.softmax(feats, dim=0)
            feat = (weights * feats).sum(dim=0)
        else:
            feat = feats.max(dim=0)[0]
            
        if self.add_residual:
            feat0 = feat[:, :self.out_channels]
            residual = F.relu(feat[:, self.out_channels:], inplace=True)
            feat = feat0 + residual
            
        return feat

class ResConv(nn.Module):
    def __init__(self, in_channels, out_channels, mixconv):
        super(ResConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Sequential( nn.Conv2d(self.in_channels, self.out_channels, 
                                                1, 1, 0, 1, 1, False),
                                    nn.BatchNorm2d(self.out_channels),
                                    nn.LeakyReLU(0.3, inplace=True)
                                  )
        self.mixconv = mixconv
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        
    def forward(self, x):
        residual = self.conv1(x)
        out = self.mixconv(residual)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = out + residual
        return out
        
class FeatLens(nn.Module):
    # if builtin_bn=True, orig_bn may be provided or None
    # if orig_bn is provided, then builtin_bn must be True
    # if orig_conv is a number, then this FeatLens instance does 
    # residual feature transformation, with 1*1 kernel
    # (before being added with the main branch features)
    # orig_conv gives the number of input channels (same as the number of output channels) 
    # if bn_only=True, FeatLens does not do convolution
    # (only learns orientation-specific batchnorms). 
    # this is an ablation study
    # to separately measure the contributions of orientation-specific batchnorms
    def __init__(self, conv_name, orig_conv, orig_bn, builtin_bn=False, 
                 num_sectors=4, num_mix_comps=10,
                 featlens_do_equalvote=True, exc_topk=5, use_softmax=True,
                 context_size=0, bn_only=False,
                 overshoot_discount=0.2, 
                 add_residual=False, 
                 two_layer_conv=False,
                 do_vote=False, is_debug=False, calc_all_exc_loss=False):
        super(FeatLens, self).__init__()

        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)        
        print("FeatLens '%s' init params:" %conv_name)
        for arg in args:
            if not isinstance(values[arg], nn.Module):
                print("%s = %s" %(arg, values[arg]))

        self.num_sectors = num_sectors
        self.two_layer_conv = two_layer_conv
        
        if type(orig_conv) is int:
            orig_in_channels = orig_conv
            self.orig_conv = nn.Sequential()
            self.in_channels = self.orig_in_channels = orig_in_channels
            self.out_channels = orig_in_channels
            # 1*1 kernel for feature transformation
            self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.has_bias = \
                (1,1), (1,1), (0,0), (1,1), 1, False 
            self.context_size = 0
            # residual features are always batchnorm-ed and relu-ed
            # simulate this behavior in the MixConv surrogate layer
            builtin_bn = True
            builtin_relu = True
        else:        
            self.in_channels = orig_conv.in_channels
            self.out_channels = orig_conv.out_channels
            self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.has_bias = \
                        orig_conv.kernel_size, orig_conv.stride, orig_conv.padding, \
                        orig_conv.dilation, orig_conv.groups, (orig_conv.bias is not None)
        
            self.context_size = context_size
            
            if orig_bn:
                self.orig_conv = nn.Sequential( orig_conv, orig_bn )
            else:
                self.orig_conv = orig_conv
            
            builtin_relu = False
            
        self.orient_convs = nn.ModuleList([self.orig_conv])
        
        self.builtin_bn = builtin_bn
        self.bn_only = bn_only
        self.do_vote = do_vote and (not self.bn_only)
        self.add_residual = add_residual
        if self.add_residual and not self.bn_only:
            self.in_channels += self.out_channels
                
        if not self.bn_only:
            for i in range(1, self.num_sectors):
                if not self.two_layer_conv:
                    mixconv = MixConv(self.in_channels, self.out_channels, 
                                            self.kernel_size, self.stride, self.padding, 
                                            self.dilation, self.groups, self.has_bias,
                                            num_comps=num_mix_comps, 
                                            use_softmax=use_softmax, 
                                            context_size=context_size, 
                                            add_residual=add_residual
                                     )

                    orient_conv = [ mixconv ]
                else:
                    mixconv = MixConv(self.out_channels, self.out_channels, 
                                            self.kernel_size, self.stride, self.padding, 
                                            self.dilation, self.groups, self.has_bias,
                                            num_comps=num_mix_comps, 
                                            use_softmax=use_softmax, 
                                            context_size=context_size, 
                                            add_residual=False
                                     )
                    orient_conv = [ ResConv(self.in_channels, self.out_channels, mixconv) ]

                if self.builtin_bn:
                    orient_conv.append(nn.BatchNorm2d(self.out_channels))
                if builtin_relu:
                    orient_conv.append(nn.ReLU(inplace=True))
                
                orient_conv = nn.Sequential(*orient_conv)
                self.orient_convs.append(orient_conv)
                
        else:
            for i in range(1, self.num_sectors):
                if type(orig_conv) is int:
                    self.orient_convs.append(nn.BatchNorm2d(self.out_channels))
                else:
                    orient_bn = nn.BatchNorm2d(self.out_channels)
                    orient_bn.load_state_dict(orig_bn.state_dict())
                    self.orient_convs.append( nn.Sequential( orig_conv, orient_bn ) )
                
        self.update_featlens = False
        self.use_featlens = False
        self.out_groudrot_featlens = False
            
        self.exc_topk = exc_topk
        self.upright_count = 0
        self.sec_idx = 0
        self.chan_pos_corrs = np.zeros(self.out_channels)
        self.chan_neg_corrs = np.zeros(self.out_channels)
        self.upd_count = 0
        self.num_iter = 0
        
        self.conv_name = conv_name
        self.sec1_angle = 360.0 / num_sectors
        self.do_equalvote = featlens_do_equalvote
        self.excite_loss = torch.cuda.FloatTensor(1)
        self.excite_loss.zero_()
        self.overshoot_discount = overshoot_discount
        self.debug = is_debug
        self.calc_all_exc_loss = calc_all_exc_loss
        self.mseloss = nn.MSELoss()
        
    def set_update_featlens(self, update_featlens):
        if not self.bn_only:
            self.update_featlens = update_featlens
        else:
            self.update_featlens = False
            
    def set_use_featlens(self, use_featlens):
        self.use_featlens = use_featlens

    # set if output the featlens conv features of the groundtruth rotation angle
    def set_out_groudrot_featlens(self, out_groudrot_featlens):
        self.out_groudrot_featlens = out_groudrot_featlens
    
    def forward(self, x):
        # angle of a counter clockwise rotation
        rot_angle = self.sec1_angle * self.sec_idx
                 
        if type(x) == list:
            x0, residual = x
            # concatenate x0 and residual features for better prediction
            if self.add_residual and not self.bn_only:
                x = torch.cat(x, dim=1)
            else:
                x = x0
                residual = 0
        else:
            x0 = x
            residual = 0
        
        # save as member vars, to be used in compute_feat_low_high()    
        self.x, self.x0, self.residual = x, x0, residual
        feat0 = None

        # try all orientations, vote for the most possible ones
        if self.use_featlens and self.do_vote:
            allsec_feat = []
            for i in range(self.num_sectors):
                if i == 0:
                    allsec_feat.append( self.orient_convs[i](x0) )
                else:
                    allsec_feat.append( self.orient_convs[i](x) )

            feat0 = allsec_feat[0]
            allsec_feat_5d = torch.stack(allsec_feat, dim=0)
            # feat_greedy is noisy and not used anywhere
            feat_greedy, orient_field_4d = allsec_feat_5d.max(dim=0)
            
            if self.do_equalvote:
                orient_field_onehot_5d = torch.cuda.FloatTensor( self.num_sectors, 
                                                                  *(orient_field_4d.shape) ).zero_()
                orient_field_onehot_5d.scatter_(0, orient_field_4d.unsqueeze(0), 1)
                # summation along the channel axis. Get the vote of all channels 
                # on each (orientation, batch, x, y)
                orient_field_chanvote = orient_field_onehot_5d.sum(dim=2)
                orient_field_chanvote_discount = orient_field_chanvote.clone()
                orient_field_chanvote_discount[0] *= 0.8
                self.orient_field_3d = orient_field_chanvote_discount.argmax(dim=0)
                orient_vote_flat = orient_field_chanvote_discount.view(self.num_sectors, -1)
                if self.use_featlens:
                    print("%s %d %03d votes: %s" %(self.conv_name, self.num_iter, 
                                rot_angle, orient_vote_flat.sum(dim=1).data.cpu().numpy()))
            else:            
                chans_total_excite = ( allsec_feat_5d * (allsec_feat_5d >=0).float() ).sum(dim=2)
                self.orient_field_3d = chans_total_excite.argmax(dim=0)
                chans_total_excite_flat = chans_total_excite.view(self.num_sectors, -1)
                if self.use_featlens:
                    print("%s %d %03d votes: %s" %(self.conv_name, self.num_iter, 
                                rot_angle, chans_total_excite_flat.sum(dim=1).data.cpu().numpy()))

            orient_indices_5d = self.orient_field_3d.unsqueeze(0).unsqueeze(2).repeat(1, 1, self.out_channels, 1, 1)
            feat_vote = torch.gather(allsec_feat_5d, 0, orient_indices_5d).squeeze(dim=0)
            if self.use_featlens:
                self.upright_count = (self.orient_field_3d == 0).sum()
            else:
                self.upright_count = x.shape[0] * x.shape[2] * x.shape[3]
                
            self.excite_loss = torch.cuda.FloatTensor(1)
            self.excite_loss.zero_()
        
        # update_featlens implies not bn_only.
        if self.update_featlens:
            # collect most excited points in the feature maps
            if self.sec_idx == 0:
                if feat0 is None:
                    feat0 = self.orient_convs[0](x0)
                    self.feat0_orig = feat0.clone()
                # try to predict the summation
                if self.add_residual:
                    feat0 += residual
                       
                feat0_cbhw = feat0.permute(1, 0, 2, 3).contiguous()
                feat0_c = feat0_cbhw.view(feat0_cbhw.shape[0], -1)
                topk_pos_excite, topk_pos_coords = feat0_c.topk(self.exc_topk, dim=1)
                self.feat0 = feat0.clone()
                self.residual = residual.clone()
                self.feat0_high = topk_pos_excite.view(-1).clone()
                self.topk_pos_coords = topk_pos_coords
                
                topk_neg_excite, topk_neg_coords = (-feat0_c).topk(self.exc_topk, dim=1)
                self.feat0_low = -topk_neg_excite.view(-1).clone()
                self.topk_neg_coords = topk_neg_coords
                high_mask_count = topk_pos_coords.numel()
                low_mask_count  = topk_neg_excite.numel()
                self.feat0_mean = feat0.mean().item()
                
                print("%s High count: %d, avg %.3f. Low count: %d, avg %.3f" % \
                            (self.conv_name, high_mask_count, self.feat0_high.mean().item(), 
                             low_mask_count, self.feat0_low.mean().item()))
            
            # compute loss. oriented kernels will be updated in the external training code
            else:
                self.compute_feat_low_high('old', do_compute_loss=True)
                
        if self.use_featlens:
            if self.out_groudrot_featlens or self.bn_only:
                if self.sec_idx == 0:
                    feat_groundrot = self.orient_convs[0](x0)
                    if self.add_residual:
                        feat_groundrot += residual
                else:
                    feat_groundrot = self.orient_convs[self.sec_idx](x)
                return feat_groundrot
            else:
                return feat_vote
        else:
            feat_orig = self.orient_convs[0](x0)
            if self.add_residual:
                feat_orig += residual
            return feat_orig
    
    def compute_feat_low_high(self, phase_name, do_compute_loss):
        rot_angle = self.sec1_angle * self.sec_idx
        if do_compute_loss:
            with torch.enable_grad():
                if self.sec_idx == 0:
                    feat_groundrot = self.orient_convs[self.sec_idx](self.x0)
                    if self.add_residual:
                        feat_groundrot = feat_groundrot + self.residual
                else:
                    feat_groundrot = self.orient_convs[self.sec_idx](self.x)
                    
                # rotate back to the original position, so 360 - angle
                feat1_invrot = rotate_tensor4d(feat_groundrot, 360 - rot_angle)
                feat1_cbhw = feat1_invrot.permute(1, 0, 2, 3).contiguous()
                feat1_c = feat1_cbhw.view(feat1_cbhw.shape[0], -1)
                feat1_orighigh_2d = torch.gather(feat1_c, 1, self.topk_pos_coords)
                feat1_orighigh = feat1_orighigh_2d.view(-1)
                feat1_origlow_2d  = torch.gather(feat1_c, 1, self.topk_neg_coords)
                feat1_origlow = feat1_origlow_2d.view(-1)
                
                with torch.no_grad():
                    high_exc_corr = pearson(self.feat0_high, feat1_orighigh)
                    low_exc_corr  = pearson(self.feat0_low,  feat1_origlow)
                    print("%s %03d %s high: %.3f/%.3f, low %.3f/%.3f, corr %.3f/%.3f" % \
                                            (self.conv_name, rot_angle, phase_name,
                                             self.feat0_high.mean().item(),
                                             feat1_orighigh.mean().item(), 
                                             self.feat0_low.mean().item(),
                                             feat1_origlow.mean().item(),
                                             high_exc_corr, low_exc_corr))

                if self.calc_all_exc_loss:
                    self.compute_allexcite_loss(self.feat0, feat1_invrot)
                else:    
                    self.compute_excite_loss(self.feat0_high, self.feat0_low, self.feat0_mean, 
                                             feat1_orighigh, feat1_origlow)
        
        else:     
            if self.sec_idx == 0:
                feat_groundrot = self.orient_convs[self.sec_idx](self.x0)
                if self.add_residual:
                    feat_groundrot = feat_groundrot + self.residual
            else:
                feat_groundrot = self.orient_convs[self.sec_idx](self.x)
                
            feat1_invrot = rotate_tensor4d(feat_groundrot, 360 - rot_angle)
            feat1_cbhw = feat1_invrot.permute(1, 0, 2, 3).contiguous()
            feat1_c = feat1_cbhw.view(feat1_cbhw.shape[0], -1)
            feat1_orighigh_2d = torch.gather(feat1_c, 1, self.topk_pos_coords)
            feat1_orighigh = feat1_orighigh_2d.view(-1)
            feat1_origlow_2d  = torch.gather(feat1_c, 1, self.topk_neg_coords)
            feat1_origlow = feat1_origlow_2d.view(-1)
                                                               
            high_exc_corr = pearson(self.feat0_high, feat1_orighigh)
            low_exc_corr  = pearson(self.feat0_low,  feat1_origlow)
            print("%s %03d %s high: %.3f/%.3f, low %.3f/%.3f, corr %.3f/%.3f" % \
                                            (self.conv_name, rot_angle, phase_name,
                                             self.feat0_high.mean().item(),
                                             feat1_orighigh.mean().item(), 
                                             self.feat0_low.mean().item(),
                                             feat1_origlow.mean().item(),
                                             high_exc_corr, low_exc_corr))
    
    def compute_allexcite_loss(self, feat0, feat1):
        allexcite_loss = self.mseloss(feat0, feat1)
        self.excite_loss = allexcite_loss * 1000
            
    def compute_excite_loss(self, feat0_high, feat0_low, feat0_mean, feat1_orighigh, feat1_origlow):
        high_diff = feat1_orighigh - feat0_high
        low_diff =  feat0_low - feat1_origlow
        feat0_high_demean = feat0_high - feat0_mean
        feat0_low_demean  = feat0_low  - feat0_mean
        low_rescale = feat0_high_demean.sum() / (0.1 - feat0_low_demean.sum()) * 2
        if low_rescale > 5.0:
            low_rescale = 5.0
            
        # low/high_underdiff: undershoot (too small when positive, or too big when negative)
        # low/high_overdiff:  overshoot  (too big when positive, or too small when negative)
        # underdiff always < 0, overdiff always > 0
        high_underdiff_mask = (high_diff < 0).float()
        high_overdiff_mask  = (high_diff > 0).float()
        low_overdiff_mask   = (low_diff > 0).float()
        low_underdiff_mask  = (low_diff < 0).float()
        high_underdiff = high_diff * high_underdiff_mask
        high_overdiff = high_diff  * high_overdiff_mask
        low_overdiff  = low_diff   * low_overdiff_mask
        low_underdiff  = low_diff  * low_underdiff_mask
        
        excite_undershoot_loss =  -(high_underdiff * feat0_high_demean).mean() - \
                                   (low_underdiff  * -feat0_low_demean).mean() * low_rescale
        excite_overshoot_loss  =   (high_overdiff  * feat0_high_demean).mean() + \
                                   (low_overdiff   * -feat0_low_demean).mean() * low_rescale

        if self.debug:
            pdb.set_trace()
                                   
        excite_loss = excite_undershoot_loss + \
                      excite_overshoot_loss * self.overshoot_discount
        excite_loss = excite_loss * 10
        # excite_loss = torch.abs(high_diff).sum() + torch.abs(low_diff).sum()
        
        avg_high_underdiff = ( high_underdiff.sum() / (high_underdiff_mask.sum() + 0.001) ).item()
        avg_high_overdiff  = ( high_overdiff.sum()  / (high_overdiff_mask.sum() + 0.001) ).item()
        avg_low_overdiff   = ( low_overdiff.sum()   / (low_overdiff_mask.sum() + 0.001) ).item()
        avg_low_underdiff  = ( low_underdiff.sum()  / (low_underdiff_mask.sum() + 0.001) ).item()
        
        print("%s lowres %.2f, h> %.3f, h< %.3f, l> %.3f, l< %.3f" %( \
                self.conv_name, low_rescale, avg_high_overdiff, avg_high_underdiff, 
                avg_low_overdiff, avg_low_underdiff))
                
        print("%s undershoot loss: %.2f, overshoot loss: %.2f" %( \
                            self.conv_name, excite_undershoot_loss.item(), 
                            excite_overshoot_loss.item() * self.overshoot_discount))
                            
        self.excite_loss = excite_loss
        