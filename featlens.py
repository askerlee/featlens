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
import copy

from PIL import Image
import math
import re
import socket
hostname = socket.gethostname()
import sys
from collections import namedtuple

np.set_printoptions(precision=4, suppress=True, threshold=sys.maxsize)
torch.set_printoptions(precision=4, sci_mode=False)

if hostname == 'm10':
    os.environ['IMAGENET'] = '/media/shaohua/ssd'
elif hostname == 'collabai31-desktop' or hostname == 'workstation2':
    os.environ['IMAGENET'] = '/data/shaohua'
else:
    print("Unknown hostname '%s'. Please specify 'IMAGENET' manually." %hostname)
    exit(0)

# benchmark the correlations of original conv feature maps. Features of
# transformed images (after inverse transformations) vs. features of original images
# CALC_ORIGFEAT_PEARSON is declared as a global variable, so the caller can update it from the outside
# CALC_ORIGFEAT_PEARSON is set to be args.calc_origfeat_pearson. Default is False
global CALC_ORIGFEAT_PEARSON
global DEBUG
global USING_MAE

# compute pearson correlation between two tensors
def pearson(t1, t2, dim=-1):
    if dim == -1:
        t1flat = t1.view(-1)
        t2flat = t2.view(-1)
        t1flatz = t1flat - t1flat.mean()
        t2flatz = t2flat - t2flat.mean()
        norm1 = (t1flatz**2).float().sum().sqrt()
        norm2 = (t2flatz**2).float().sum().sqrt()
        norm1[norm1 < 1e-5] = 1
        norm2[norm2 < 1e-5] = 1

        corr = (t1flatz * t2flatz).float().sum() / (norm1 * norm2)
        return corr.item()

    elif dim == 0:
        t1flat = t1.view(t1.shape[0], -1)
        t2flat = t2.view(t2.shape[0], -1)
        t1flatz = t1flat - t1flat.mean(dim=1, keepdim=True)
        t2flatz = t2flat - t2flat.mean(dim=1, keepdim=True)
        norm1 = torch.pow(t1flatz, 2).float().sum(dim=1).sqrt()
        norm2 = torch.pow(t2flatz, 2).float().sum(dim=1).sqrt()
        norm1[norm1 < 1e-5] = 1
        norm2[norm2 < 1e-5] = 1

        corr = (t1flatz * t2flatz).float().sum(dim=1) / (norm1 * norm2)
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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeters(object):
    """Computes and stores the average and current values of given keys"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = {}
        self.avg = {}
        self.sum = {}
        self.count = {}

    def update(self, key, val, n=1):
        self.val[key] = val
        self.sum.setdefault(key, 0)
        self.count.setdefault(key, 0)
        self.avg.setdefault(key, 0)
        self.sum[key] += val * n
        self.count[key] += n
        self.avg[key] = self.sum[key] / self.count[key]

def print_maxweight(model):
    weight_max, weight_min = 0, 0
    max_param_name, min_param_name = None, None
    for name, weight in model.named_parameters():
        if 'weight' in name:
            if weight.max().item() > weight_max:
                weight_max = weight.max().item()
                max_param_name = name
            if weight.min().item() < weight_min:
                weight_min = weight.min().item()
                min_param_name = name

    print( "max '%s' %.4f, min '%s' %.4f" %( \
            max_param_name, weight_max, min_param_name, weight_min) )


def get_pos_neg_stats(feat):
    # detach() only changes requires_grad, doesn't make a copy
    feat = feat.detach()
    feat_pos = torch.masked_select(feat, feat > 0)
    feat_pos_std, feat_pos_mean = torch.std_mean(feat_pos, unbiased=False)
    feat_neg = torch.masked_select(feat, feat < 0)
    feat_neg_std, feat_neg_mean = torch.std_mean(feat_neg, unbiased=False)
    return feat_pos_mean.item(), feat_neg_mean.item(), \
           feat_pos_std.item(),  feat_neg_std.item()

# Channel-wise mean/std of positive and negative feature values, respectively
# chan_*_mean, chan_*_std: require_grads = False
def get_chan_pos_neg_stats(feat):
    feat = feat.detach()
    feat_cbhw = feat.permute(1, 0, 2, 3).contiguous()
    feat_c = feat_cbhw.view(feat_cbhw.shape[0], -1)
    pos_mask = (feat_c > 0).float()
    neg_mask = (feat_c < 0).float()
    feat_pos = feat_c * pos_mask
    feat_neg = feat_c * neg_mask
    num_pos = pos_mask.sum(dim=1) + 0.0001
    num_neg = neg_mask.sum(dim=1) + 0.0001
    chan_pos_mean = feat_pos.sum(dim=1) / num_pos
    chan_neg_mean = feat_neg.sum(dim=1) / num_neg
    
    chan_pos_std = torch.sqrt( (feat_pos*feat_pos).sum(dim=1) / num_pos - chan_pos_mean*chan_pos_mean )
    chan_neg_std = torch.sqrt( (feat_neg*feat_neg).sum(dim=1) / num_neg - chan_neg_mean*chan_neg_mean )
    return chan_pos_mean, chan_neg_mean, chan_pos_std, chan_neg_std
    
def get_dict_stratum(d, stratum_dindices):
    d2 = {}
    for k in d:
        d2[k] = d[k][stratum_dindices]
    return d2
    
# a pass-through class
class identity(nn.Module):
    def forward(self, x):
        return x

class RotateTensor4d(nn.Module):
    def __init__(self, angle):
        super(RotateTensor4d, self).__init__()
        self.angle = angle
    def forward(self, x):
        if self.angle == 0 or self.angle == 360:
            rot_x = x
        elif self.angle == 90:
            rot_x = x.transpose(2, 3).flip(2)
        elif self.angle == 180:
            rot_x = x.flip(2).flip(3)
        elif self.angle == 270:
            rot_x = x.transpose(2, 3).flip(3)
        else:
            raise NotImplementedError

        return rot_x

    def __str__(self):
        return "RotateTensor4d(%d)" %self.angle

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

class ScaleTensor4d(nn.Module):
    def __init__(self, scale_h, scale_w):
        super(ScaleTensor4d, self).__init__()
        self.scale_h = scale_h
        self.scale_w = scale_w

    def forward(self, x):
        h, w = x.shape[2:]
        h2 = round(h * self.scale_h)
        w2 = round(w * self.scale_w)
        x2 = nn.functional.interpolate(x, (h2, w2), mode='bilinear', align_corners=True)

        return x2

    def __str__(self):
        return "ScaleTensor4d(%g,%g)" %(self.scale_h, self.scale_w)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# MixConv is primarily used for rotation lenses.
# MixConv can also be used as the first layer of MixConv2L.
# If used in MixConv2L, always sim_residual=False.
# Only if MixConv is used directly to simulate the features,
# (without being followed by a second layer), then sim_residual=True
# output pre-relu features
class MixConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, groups, num_comps=5, using_softmax_attn=True,
                 context_size=0, sim_residual=False):
        super(MixConv, self).__init__()
        self.K = num_comps
        self.using_softmax_attn = using_softmax_attn and self.K > 1
        self.in_channels, self.out_channels, self.orig_kernel_size, self.stride, \
            self.orig_padding, self.dilation, self.groups = \
                in_channels, out_channels, kernel_size, stride, \
                    padding, dilation, groups

        self.padding = (self.orig_padding[0] + context_size, self.orig_padding[1] + context_size)
        self.kernel_size = (self.orig_kernel_size[0] + 2 * context_size,
                            self.orig_kernel_size[1] + 2 * context_size)

        self.sim_residual = sim_residual

        if self.sim_residual:
            self.C = 2
        else:
            self.C = 1

        # num_comps mixture of conv. Each conv has out_channels * C * K channels
        # if used in MixConv2L, C = 1
        self.mix_conv = nn.Conv2d(self.in_channels, self.out_channels * self.C * self.K,
                                  self.kernel_size, self.stride, self.padding,
                                  self.dilation, self.groups, bias=True) # always use bias
        torch.nn.init.xavier_uniform_(self.mix_conv.weight)

    def forward(self, x):
        if type(x) == dict:
            if self.sim_residual:
                x_cat = torch.cat((x['pre_conv'], x['shortcut']), dim=1)
            else:
                x_cat = x['pre_conv']
        # otherwise x is tensor. no conversion needed
        elif self.sim_residual:
            # BUG: cannot simulate residual connection on a tensor
            pdb.set_trace()

        # split to self.K (num_comps) groups of features, and aggregate them with softmax attn
        feats = torch.split(self.mix_conv(x_cat), self.out_channels * self.C, dim=1)
        feats = torch.stack(feats, dim=0)

        if self.using_softmax_attn:
            weights = torch.softmax(feats, dim=0)
            feat = (weights * feats).sum(dim=0)
        else:
            feat = feats.max(dim=0)[0]

        if self.sim_residual:
            feat_main = feat[:, :self.out_channels]
            residual = F.relu_(feat[:, self.out_channels:])
            feat = feat_main + residual

        if DEBUG:
            pdb.set_trace()
            
        return feat

# MixConv2L is primarily used for scaling lenses.
# output pre-relu features
class MixConv2L(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride, padding,
                 num_comps=5, using_softmax_attn=True, sim_residual=False,
                 interp_origfeat=False):
        super(MixConv2L, self).__init__()
        self.mix_convs = nn.ModuleList()
        self.K = num_comps
        self.using_softmax_attn = using_softmax_attn and self.K > 1
        self.in_channels, self.mid_channels, self.out_channels, self.kernel_size, \
            self.stride, self.padding = \
                in_channels, mid_channels, out_channels, kernel_size, \
                stride, padding

        self.sim_residual = sim_residual

        if self.sim_residual:
            self.C = 2
        else:
            self.C = 1

        print("MixConv2L: in_chan: %d, mid_chan: %d, out_chan: %d*%d" % \
                (in_channels, mid_channels, out_channels, self.C))

        self.interp_origfeat = interp_origfeat
        if self.interp_origfeat:
            # Fixed weights. Effectively disable lens, 
            # and use bilinear-interpolated features only
            # CALC_ORIGFEAT_PEARSON should only be used during test phase for computing stats.
            # Otherwise scaling lenses won't get optimal performance
            if CALC_ORIGFEAT_PEARSON:
                self.interp_weight = torch.tensor([0, 1])
            else:
                # trainable weights
                self.interp_weight = nn.Parameter(torch.tensor([0.5, 0.5]), requires_grad=True)
        else:
            self.interp_weight = [1.0, 0.0]
        self.interp_weight_range = (1.0, 2.0)

        trans_conv = []
        trans_conv.append( MixConv(self.in_channels, self.mid_channels,
                            (1,1), (1,1), (0,0), 1, 1,
                            num_comps=self.K, using_softmax_attn=True,
                            context_size=0, sim_residual=False) )
        trans_conv.append( nn.BatchNorm2d(self.mid_channels) )
        trans_conv.append( nn.ReLU(inplace=True) )
        trans_conv.append( nn.ConvTranspose2d(self.mid_channels, self.out_channels,
                                                self.kernel_size, self.stride, self.padding,
                                                groups=1, bias=True) ) # always use bias

        trans_conv = nn.Sequential(*trans_conv)
        # torch.nn.init.xavier_uniform_(trans_conv[3].weight)
        self.trans_conv = trans_conv
        self.bn = nn.BatchNorm2d(self.out_channels, affine=True)

    def forward(self, x):
        # if sim_residual=True, then x = (x0, residual, orig_feat)
        # else x = (x0, orig_feat)
        if self.sim_residual:
            x_cat = torch.cat( [ x['pre_conv'], x['shortcut'] ], dim=1 )
        else:
            x_cat = x['pre_conv']

        # trans_conv always takes a tensor as input
        feat = self.trans_conv(x_cat)
        # x['orig_feat'] are the original features of the transformed images.
        # detach() to cut off the gradients, avoid accidentally updating the original features
        x_orig = x['orig_feat'].detach()

        if self.interp_origfeat:
            # normalize interp_weight first
            with torch.no_grad():
                self.interp_weight[self.interp_weight < 0.0001] = 0.0001
                if self.interp_weight.sum() > self.interp_weight_range[1]:
                    self.interp_weight.mul_( self.interp_weight_range[1] / self.interp_weight.sum() )
                elif self.interp_weight.sum() < self.interp_weight_range[0]:
                    self.interp_weight.mul_( self.interp_weight_range[0] / self.interp_weight.sum() )

            interp_x_orig = nn.functional.interpolate(x_orig, feat.shape[2:],
                                                      mode='bilinear', align_corners=True)
        else:
            interp_x_orig = 0

        # interp_x_orig has been BN'ed, so feat has to be BN'ed as well
        # bn(feat) to align the distribution of feat and interp_x_orig
        # in the resnet residual connection, residual has been relu'ed
        # but doing relu(interp_x_orig) seems to hurt performance
        feat = self.bn(feat) * self.interp_weight[0] + \
               interp_x_orig * self.interp_weight[1]

        return feat

# output pre-relu features
class OrigConv(nn.Module):
    def __init__(self, name, orig_conv, orig_bn, sim_residual):
        super(OrigConv, self).__init__()
        self.name = name
        # orig_bn could be None, i.e. no BN in the original conv layer
        self.bn = copy.deepcopy(orig_bn)
        self.conv = copy.deepcopy(orig_conv)
        self.sim_residual = sim_residual

    def forward(self, x):
        if self.sim_residual:
            feat = self.conv(x['pre_conv'])
            if self.bn:
                feat = self.bn(feat)
            feat = feat + x['shortcut']
        else:
            feat = self.conv(x['pre_conv'])
            if self.bn:
                feat = self.bn(feat)
        return feat

# do rotation classification
class RotClassifier(nn.Module):
    def __init__(self, sim_residual, num_rot_lenses, rot_feat_shape, avgpool_size=None):
        super(RotClassifier, self).__init__()
        self.sim_residual = sim_residual
        self.num_rot_lenses = num_rot_lenses
        self.rot_feat_shape = rot_feat_shape
        self.avgpool_size = avgpool_size
        if self.avgpool_size:
            self.rot_feat_num = rot_feat_shape[0] * self.avgpool_size[0] * self.avgpool_size[1]
            self.pooler = nn.AdaptiveAvgPool2d(self.avgpool_size)
        else:
            self.rot_feat_num = rot_feat_shape[0] * rot_feat_shape[1] * rot_feat_shape[2]
        self.cls = nn.Linear(self.rot_feat_num, self.num_rot_lenses)

    def forward(self, x):
        if self.sim_residual:
            x_cat = torch.cat( [x['pre_conv'], x['shortcut']], dim=1 )
        else:
            x_cat = x['pre_conv']

        if self.avgpool_size:
            x_cat = self.pooler(x_cat)

        x_cat = x_cat.view(x_cat.shape[0], -1)
        scores = self.cls(x_cat)
        return scores

def get_top_activations(feat, act_topk):
    feat_cbhw = feat.permute(1, 0, 2, 3).contiguous()
    # co: channel as 1d, and all other dimensions flattened as 1d
    feat_co = feat_cbhw.view(feat_cbhw.shape[0], -1)
    feat_top_high, feat_top_high_coords = feat_co.topk(act_topk, dim=1)
    # feat_high is 2d: channel, top-k highest activations
    feat_high = feat_top_high.detach().clone()

    feat_top_low, feat_top_low_coords = (-feat_co).topk(act_topk, dim=1)
    # feat_low is 2d: channel, top-k lowest activations
    feat_low = -feat_top_low.detach().clone()
    return feat_co, feat_high, feat_top_high_coords, feat_low, feat_top_low_coords

class StrataIterator:
    def __init__(self, lenskit):
        self.lenskit = lenskit
        
    def __iter__(self):
        self.all_strata_lens_indices  = self.lenskit.all_strata_lens_indices
        self.all_lens_strata_dindices = self.lenskit.all_lens_strata_dindices
        self.stratum_iter_idx = 0
        self.total_strata_num = len(self.all_strata_lens_indices)
        return self
    
    def __next__(self):
        if self.stratum_iter_idx < self.total_strata_num:
            self.lenskit.stratum_lens_idx = self.all_strata_lens_indices[self.stratum_iter_idx]
            self.lenskit.lens_stratum_dindices = self.all_lens_strata_dindices[self.lenskit.stratum_lens_idx]
            self.lenskit.update_topk_num()
            self.stratum_iter_idx += 1
            if self.lenskit.dindices_are_stratified:
                self.lenskit.curr_lens_name = 's%d' %self.lenskit.stratum_lens_idx
            # if not stratified, keep the assigned lens name from the caller

            return self.stratum_iter_idx - 1
        else:
            raise StopIteration
        
# Each LensKit contains multiple Lens
class LensKit(nn.Module):
    # orig_conv is provided to get the conv parameters.
    # when bn_only, it can be used to make multiple conv layers with lens-specific batchnorms
    # if builtin_bn=True, each lens has a BN
    # if bn_only=True, LensKit does not do convolution (only learns lens-specific batchnorms).
    # this is an ablation study to separate the contributions of lens-specific batchnorms
    # sim_residual: simulate the sum of the original features and residual features. default: True
    def __init__(self, host_stage_name, orig_conv, orig_bn,
                 # a list of [ lens_name, img_trans, feat0_transfun, inv_trans, act_topk_discount ]
                 lenses_spec,
                 builtin_bn=True,
                 num_mix_comps=5,
                 base_act_topk=600,     # topk activations (both positive and negative) to be aligned
                 top_act_weighted_by_feat0=True,
                 overshoot_discount=0.2,
                 using_softmax_attn=True, # use softmax to compute attention weights in MixConv
                 context_size=0, # context size when aligning features. Not recommended to use > 0.
                 bn_only=False,
                 sim_residual=True,
                 doing_post_relu=True, # relu the features before returning them
                 using_rot_lenses=True,
                 guessing_rot_lenses=True,
                 rot_cls_avgpool_size=None,
                 num_rot_lenses=4, # counting in the original lens
                 rot_feat_shape=(2560, 7, 7), # used to initialize the RotClassifier
                 rot_idx_mapper=None,
                ):
        super(LensKit, self).__init__()

        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        print("LensKit '%s' init params:" %host_stage_name)
        for arg in args:
            if not isinstance(values[arg], nn.Module):
                print("%s = %s" %(arg, values[arg]))

        self.num_lenses = len(lenses_spec)
        self.all_lenses = []

        self.in_channels = orig_conv.in_channels
        self.out_channels = orig_conv.out_channels
        self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.has_bias = \
                    orig_conv.kernel_size, orig_conv.stride, orig_conv.padding, \
                    orig_conv.dilation, orig_conv.groups, (orig_conv.bias is not None)

        self.context_size = context_size

        self.builtin_bn = builtin_bn
        self.bn_only = bn_only
        self.sim_residual = sim_residual
        if self.sim_residual and not self.bn_only:
            self.in_channels += self.out_channels

        # there's always no relu in orig_conv. relu is done in the host model
        # orig_conv is only for debugging purposes. It shouldn't be used to generate orig_feat0
        self.orig_conv = OrigConv('orig', orig_conv, orig_bn, self.sim_residual)

        # all_lenses[0] is fixed to be an empty, do-nothing Sequential. 
        # But it should never be used, as the input is a dictionary self.x,
        # and the output features are expected to be a tensor
        self.lens0 = nn.Sequential()
        self.lens0.name = 'orig'
        self.all_lenses = nn.ModuleList([self.lens0])

        self.topk_discounts = [ 1 for i in range(self.num_lenses) ]
        # transforming function of each lens
        self.all_feat0_transfun = [ None for i in range(self.num_lenses) ]
        # shape of feature tensors fed to different lenses in the current data iteration
        self.feat_shapes = [ None for i in range(self.num_lenses) ]
        # inv_transforms: the collection of custom post-processing transformations of all lenses
        self.inv_transforms = [None]

        if not self.bn_only:
            # lenses_spec[0] is a placeholder for orig_conv. skip it here
            for i in range(1, self.num_lenses):
                lens_name, img_trans, feat0_transfun, inv_trans, topk_discount = lenses_spec[i]
                self.topk_discounts[i] = topk_discount
                self.all_feat0_transfun[i] = feat0_transfun

                # inv_trans: either a list of TransConv params of the lens, 
                # or a custom post-lens processing function
                # For scaling transformations, the lens transformation is always a 2-layer MixConv2L
                # In this case, inv_trans provides the MixConv2L params.
                # For rotation, inv_trans is an inverse rotation function provided by the caller 
                # to do post-lens processing.
                # If inv_trans is a custom function (such as for rotations), 
                # the lens is a 1-layer MixConv by default
                if type(inv_trans) == list:
                    # 2048, (1,1), (1,1), (0,0), True
                    mid_channels, kernel_size, stride, padding, interp_origfeat = inv_trans
                    use_conv2L = True
                    self.inv_transforms.append(None)
                else:
                    kernel_size, stride, padding = self.kernel_size, self.stride, self.padding
                    use_conv2L = False
                    self.inv_transforms.append(inv_trans)

                if not use_conv2L:
                    # in_channels: 2560 (512 conv2 + 2048 shortcut), out_channels: 2048.
                    mixconv = MixConv(self.in_channels, self.out_channels,
                                            kernel_size, stride, padding,
                                            self.dilation, self.groups,
                                            num_comps=num_mix_comps,
                                            using_softmax_attn=using_softmax_attn,
                                            context_size=context_size,
                                            sim_residual=sim_residual
                                     )
                else:
                    # in_channels: 2560 (512 conv2 + 2048 shortcut), out_channels: 2048, 
                    # mid_channels: 2048.
                    mixconv = MixConv2L(self.in_channels, self.out_channels,
                                            mid_channels, kernel_size, stride, padding,
                                            num_comps=num_mix_comps,
                                            using_softmax_attn=using_softmax_attn,
                                            sim_residual=sim_residual,
                                            interp_origfeat=interp_origfeat,
                                        )

                lens = [ mixconv ]

                if self.builtin_bn:
                    lens.append(nn.BatchNorm2d(self.out_channels, affine=True))

                # inv_trans is a module to inversely transform new features
                # back to geometrically align with self.feat0
                # if use_conv2L, inv_trans is the specs of MixConv2L
                # otherwise, it's a custom function provided by the caller
                if not use_conv2L:
                    lens.append(inv_trans)
                lens = nn.Sequential(*lens)
                lens.name = lens_name
                self.all_lenses.append(lens)

        else:
            # lenses_spec[0] is a placeholder for orig_conv. skip it here
            for i in range(1, self.num_lenses):
                # inv_trans is not used, as orig_conv is used in each lens
                # features are of the original size, alignment is not necessary
                lens_name, img_trans, feat0_transfun, inv_trans, topk_discount = lenses_spec[i]

                lens = OrigConv(lens_name, orig_conv, orig_bn, self.sim_residual)
                # the only difference between different lenses is BN statistics
                self.all_lenses.append(lens)

        self.using_rot_lenses = using_rot_lenses
        if self.using_rot_lenses:
            self.num_rot_lenses = num_rot_lenses
            self.guessing_rot_lenses = guessing_rot_lenses
            # Even if not guessing_rot_lenses, rot_cls is still here, but not called.
            # This is to make the dump of guessing_rot models and ground rot models compatible
            self.rot_cls = RotClassifier(sim_residual=sim_residual, num_rot_lenses=num_rot_lenses,
                                         rot_feat_shape=rot_feat_shape, 
                                         avgpool_size=rot_cls_avgpool_size)
            self.rot_idx_mapper = rot_idx_mapper
        else:
            self.num_rot_lenses = 0
            self.guessing_rot_lenses = False

        # train: update, test: freeze
        # if bn_only, never update lenses (only update BN stats implicitly)
        self.updating_lens = False
        self.using_lenses = False
        # frozen: do not update lenses
        self.frozen = True
        self.phase_name = None

        self.feat0_base_act_topk = base_act_topk
        self.feat1_base_act_topk = int(base_act_topk / 2)
        self.top_act_weighted_by_feat0 = top_act_weighted_by_feat0
        self.upright_count = 0
        self.chan_pos_corrs = np.zeros(self.out_channels)
        self.chan_neg_corrs = np.zeros(self.out_channels)
        self.upd_count = 0

        self.host_stage_name = host_stage_name
        self.overshoot_discount = overshoot_discount
        self.mseloss = nn.MSELoss()
        self.maeloss = nn.L1Loss()
        
        if doing_post_relu:
            self.postproc = nn.ReLU(inplace=False)
        else:
            self.postproc = nn.Sequential()

        self.strata_iter_obj = StrataIterator(self)
        self.clear_cache()
        # dindices: data indices, i.e., indices to input images, grouped by ground lens
        self.all_ground_lens_strata_dindices = [ None for i in range(self.num_lenses) ]
        
    def clear_cache(self):
        self.feat0 = None
        self.feat0_chan_pos_mean, self.feat0_chan_neg_mean, self.feat0_chan_pos_std, self.feat0_chan_neg_std = 0, 0, 0, 0
        # orig_feat0 are the original CNN features before lens transformation of the original images
        self.orig_feat0 = None
        self.feat0_2d, self.feat0_high, self.feat0_low, self.feat0_high_coords, \
            self.feat0_low_coords = None, None, None, None, None
        self.meanfeat_corr, self.allfeat_corr = 0, 0
        # Indices of all ground lens whose corresponding images appear in the input.
        # if in the 'strata' stage on 'mnist', all_ground_lens_indices always includes all lenses. 
        # But it's possible some lenses don't have corresponding images. 
        # No need to exclude these lenses from all_ground_lens_indices
        self.all_ground_lens_indices = []
        self.all_lens_strata_dindices = [ None for i in range(self.num_lenses) ]
        # a list of indices of lenses to be iterated when input images are strata
        # belonging to different lenses that should be transformed respectively
        self.all_strata_lens_indices = []
        # index of the current lens (may be guessed, and may be different from the ground lens)
        self.stratum_lens_idx = 0
        self.lens_stratum_dindices = None
        self.dindices_are_stratified = False
        with torch.enable_grad():
            self.whole_activ_loss = torch.tensor(0., requires_grad=True).cuda()
            self.top_activ_loss   = torch.tensor(0., requires_grad=True).cuda()
        # the losses are not computed for ground lens, so they are always 0
        self.all_lenses_whole_activ_loss = {0: 0}
        self.all_lenses_top_activ_loss = {0: 0}
        self.rot_scores = None
        
    def guess_rot_lens(self, x):
        # guess_rot_lens() only be called if (self.using_rot_lenses and self.guessing_rot_lenses)
        # in our simplified setting, guessing_rot_lenses implies this lenskit only contains rotation lenses
        with torch.enable_grad():
            self.rot_scores = self.rot_cls(x)
        guessed_rot_labels = self.rot_scores.argmax(dim=-1)
        # in case the prediction classes are indexed differently from the rotation lenses
        # but if the rotation lenses are the first N lenses (lens 0 is the original lens), 
        # then they have the same indices, and rot_idx_mapper=None
        if self.rot_idx_mapper:
            guessed_rot_labels = torch.tensor([ self.rot_idx_mapper[rot_idx] for rot_idx in guessed_rot_labels ]).cuda()

        guessed_all_rot_strata_dindices = []
        for i in range(self.num_rot_lenses):
            rot_data_indices = torch.masked_select( torch.arange(len(guessed_rot_labels)).cuda(), guessed_rot_labels==i )
            guessed_all_rot_strata_dindices.append(rot_data_indices)

        self.guessed_rot_labels = guessed_rot_labels
        self.guessed_all_rot_strata_dindices = guessed_all_rot_strata_dindices

    def update_topk_num(self):
        topk_discount = self.topk_discounts[self.stratum_lens_idx]
        lens_stratum_batch_size = len(self.lens_stratum_dindices)
        # Discount the K of top-K activations we should optimize on
        self.feat0_act_topk = round( self.feat0_base_act_topk * lens_stratum_batch_size * topk_discount )
        self.feat1_act_topk = round( self.feat1_base_act_topk * lens_stratum_batch_size * topk_discount )
            
    def forward(self, x):
        # 'pre_conv' and 'pre_relu' should be fed in by all host models
        # 'shortcut' is only available in ResNet
        self.input_feat_shape = list(x['pre_conv'].shape)
        if self.sim_residual:
            self.input_feat_shape[1] += x['shortcut'].shape[1]

        self.x = {}
        self.x['pre_conv'] = x['pre_conv']
        # 'out' is the original final features after relu
        self.x['out'] = x['out']
        if self.sim_residual:
            self.x['shortcut'] = x['shortcut']
        
        # x['orig_feat'] are the original features of the transformed images,
        # not the same as orig_feat0, which are the original features of the original images.
        # orig_feat0 is unavailable during test phase.
        self.x['orig_feat'] = x['pre_relu']
        # the lens loop starts with input_lens_iter_idx=0, which means all input are original images. 
        # So feat0 always contains assigned value through the lens loop
        if self.input_lens_iter_idx == 0:
            # has to keep both orig_feat0 (grad version) and feat0 (no grad version)
            # orig_feat0 is used to keep the gradient flow of the original model
            self.orig_feat0 = x['pre_relu']
            # orig_feat0 is possible to be accidentally changed by the host model
            # (which may happen if doing_post_relu=False and host model
            # does inplace ReLU on returned features),
            # But feat0 has no gradient, and won't be accidentally changed by BP.
            # So the original features are unchanged in feat0
            self.feat0 = self.orig_feat0.detach().clone()
        # Otherwise x is strata for different lenses

        # The flag dindices_are_stratified is set by the external data feeding code
        # For 'mnist', it first has a ground lens iteration first.
        # In this iteration, all images are original, thus dindices_are_stratified = False
        # It's followed by an iteration with all_ground_lens_indices == [0..N-1], 
        # i.e. strata of images for different lenses, so dindices_are_stratified = True.
        # For ImageNet, it first has a ground lens iteration 
        # (input_lens_iter_idx = 0, all_ground_lens_indices = [0]), 
        # followed by iterations with all_ground_lens_indices == [i] (i in {1, ..., N-1}), 
        # and all_ground_lens_strata_dindices is not strata of images.
        # But if using_guessed_lenses, guessed_all_rot_strata_dindices may be strata (due to classification errors)
        if self.using_rot_lenses and self.guessing_rot_lenses:
            self.guess_rot_lens(self.x)
            if self.using_guessed_lenses:
                # In test phase non-original lenses iterations, enumerate all lenses, 
                # use the corresponding predicted lens_stratum_dindices to select a subset of images.
                # Then use the rotation lens indexed by stratum_lens_idx to do feature transformations.
                # Strata of features are assigned to feat_all_lenses indexed by lens_stratum_dindices
                # before returning to the host model.
                # As using_guessed_lenses, guessed_all_rot_strata_dindices could be a mixture (due to classification errors)
                # So all_strata_lens_indices = [0, ..., N-1]
                self.all_strata_lens_indices  = list(range(self.num_lenses))
                self.all_lens_strata_dindices = self.guessed_all_rot_strata_dindices
            else:
                # In training phase or test phase's original lens iterations (stratum_lens_idx==0), 
                # use all_ground_lens_strata_dindices to select images, 
                # (all images on ImageNet, subsets of images on mnist)
                # and extract features or do feature transformations,
                # In training phase, optimize self.rot_cls w.r.t. the rotation lenses prediction loss.
                self.all_strata_lens_indices  = self.all_ground_lens_indices
                self.all_lens_strata_dindices = self.all_ground_lens_strata_dindices
        else:
            self.all_strata_lens_indices  = self.all_ground_lens_indices
            self.all_lens_strata_dindices = self.all_ground_lens_strata_dindices

        feat_all_lenses = torch.zeros_like(self.feat0)
        strata_iterator = iter(self.strata_iter_obj)
        # Reset accumulated losses, as we'll add lens-wise losses to 
        # the accumulated losses in the strata lens iteration below.
        # requires_grad=True, so that when all_strata_lens_indices = [0], 
        # activ_loss=tensor(0.), and activ_loss.backward() won't yield an exception
        with torch.enable_grad():
            self.whole_activ_loss = torch.tensor(0., requires_grad=True).cuda()
            self.top_activ_loss   = torch.tensor(0., requires_grad=True).cuda()
        
        self.all_lens_strata_batch_sizes = np.zeros(self.num_lenses, dtype=int)
        # self.stratum_lens_idx and self.lens_stratum_dindices are set and updated by strata_iterator / strata_iter_obj
        # in the strata lens iteration below. stratum_lens_idx iterates all_strata_lens_indices
        # stratum_iter_idx is different from stratum_lens_idx: stratum_iter_idx is the iteration index to all_strata_lens_indices,
        # stratum_lens_idx is an element in all_strata_lens_indices
        for stratum_iter_idx in strata_iterator:
            self.all_lenses_whole_activ_loss[self.stratum_lens_idx] = 0
            self.all_lenses_top_activ_loss[self.stratum_lens_idx] = 0
            lens_stratum_batch_size = len(self.lens_stratum_dindices)
            self.all_lens_strata_batch_sizes[self.stratum_lens_idx] = lens_stratum_batch_size
            if lens_stratum_batch_size == 0:
                continue
                
            # when "not frozen" (not eval_only) and training, get_top_activations()
            # is mandatory for computing losses.
            # During a "not frozen" test phase, get_top_activations() is necessary for later alignment
            # optimization ( top activations on original_data <=> lens(transformed_data) ).
            # Top activations are cached and used in the subsequent training phase
            # Besides, we compute feature correlations to see how it improves due to training
            # (test happens before training, so correlations in training phase should be larger)
            # when eval_only, we still want to get feature correlations for bookkeeping
            # but as lenses are frozen, training/test correlations are the same. only compute once
            getting_top_act = (not self.frozen) or (self.phase_name == 'test')
            if getting_top_act or DEBUG:
                # Collect and cache most activated points in the feature maps feat0.
                if self.input_lens_iter_idx == 0:
                    self.feat0_chan_pos_mean, self.feat0_chan_neg_mean, self.feat0_chan_pos_std, \
                        self.feat0_chan_neg_std = get_chan_pos_neg_stats(self.feat0)
                        
                    feat0_2d, feat0_high, feat0_high_coords, feat0_low, \
                        feat0_low_coords = get_top_activations(self.feat0, self.feat0_act_topk)

                    # Cache top activations to be used during the training phase (reusable across lenses)
                    self.feat0_high = feat0_high
                    self.feat0_high_coords = feat0_high_coords
                    self.feat0_low = feat0_low
                    self.feat0_low_coords = feat0_low_coords
                    high_mask_count = feat0_high.numel()
                    low_mask_count  = feat0_low.numel()
                    self.feat0_2d = feat0_2d

                    # updating_lens: (not self.frozen) and phase_name == 'train'.
                    # (frozen = eval_only) always holds.
                    # Note that even if frozen, we may still enter training phase,
                    # to evaluate performance without lens.
                    # (In training phase does both training and evaluation of original performance. 
                    # But if frozen, no training and updating_lens, only evaluation)
                    if self.updating_lens:
                        print("%s High count: %d, avg %.3f. Low count: %d, avg %.3f" % \
                                (self.curr_lens_name, high_mask_count, self.feat0_high.mean().item(),
                                 low_mask_count, self.feat0_low.mean().item()))

                # Compute correlations and losses. 
                # If updating_lens, lense will be updated in the external training code.
                # updating_lens => not bn_only (bn_only requires no param update).
                # If frozen, no need to compute losses, which are primarily used for BP.
                # (also no point to compare losses between training/test)
                # If the current lens is the original lens, no need to contrast_feat_low_high() 
                # self.input_lens_iter_idx == 0 implies self.stratum_lens_idx == 0, 
                # so it's enough only checking stratum_lens_idx == 0.
                if (not self.frozen) and self.stratum_lens_idx != 0:
                    self.contrast_feat_low_high('old', do_compute_loss=True, verbose=True)

            # Current lens is the original lens. Store original untouched model features.
            # Note lens_stratum_dindices may be guessed and may contain errors.
            # Use orig_feat0 instead of feat0 to retain gradients
            # (If orig_feat0 has. feat0 always has no gradients)
            if self.stratum_lens_idx == 0:
                lens0_feat = self.orig_feat0[self.lens_stratum_dindices]
                feat_all_lenses.index_copy_(0, self.lens_stratum_dindices, lens0_feat)
                continue
                
            # during training, using_lenses = False
            if self.using_lenses:
                # if stratum_lens_idx == 0, cannot reach here (has returned above). so stratum_lens_idx > 0
                # allfeat_currlens: features of all image processed by the current lens.
                # i.e., the lens for the transformation applied to the current subset of images,
                # such as a particular scaling or rotation.
                # feat_currlens: selected subset of features processed by the current lens.
                # A more efficient way is to select image features first, and then go through the current lens
                # The way we chose is easier to implement, as we don't muddle up with different arrays in self.x
                x_stratum = get_dict_stratum(self.x, self.lens_stratum_dindices)
                feat_stratum = self.all_lenses[self.stratum_lens_idx](x_stratum)
                self.feat_shapes[self.stratum_lens_idx] = list(feat_stratum.shape)
                feat_all_lenses.index_copy_(0, self.lens_stratum_dindices, feat_stratum)
            else:
                # Separately assign feat_shapes of each lens
                self.feat_shapes[self.stratum_lens_idx] = list(x['out'].shape)
                self.feat_shapes[self.stratum_lens_idx][0] = len(self.lens_stratum_dindices)
            
        # end of strata lens loop

        # test phase
        if self.using_lenses:
            # Before feat_all_lenses are returned to host model, it's postprocessed
            # by a *non-inplace* ReLU in postproc, so its contents shouldn'be be accidentally changed
            return self.postproc(feat_all_lenses)
        
        # training phase    
        else:
        # Since not using_lenses, just pass through x['out'].
        # Return the orig_out_feat (corresponding to different lenses) as a whole.
        # No need to separate into individual features and then re-assemble them.
        # x['out'] is already relu'ed. No need to postproc()
            orig_out_feat = x['out']
            return orig_out_feat

    # contrast original features with features of transformed images
    # (whether after applying lens or not)
    def contrast_feat_low_high(self, compfeat_phase_name, do_compute_loss, verbose=False):
        if do_compute_loss:
            with torch.enable_grad():
                # stratum_lens_idx == 0 should never happen here. 
                # In this case, contrast_feat_low_high() is skipped
                # feat1 contains features of all images. Will select a stratum later
                if self.stratum_lens_idx == 0:
                    feat1 = self.feat0
                else:
                    if self.dindices_are_stratified:
                        x_stratum = get_dict_stratum(self.x, self.lens_stratum_dindices)
                    else:
                        x_stratum = self.x
                    feat1 = self.all_lenses[self.stratum_lens_idx](x_stratum)

                # feat0_transfun is a transformation function to feat0
                # by default feat0_transfun = None
                # feat0_transfun is specified among origsize_lenses 
                # (doing downsampling to original features)
                feat0_transfun = self.all_feat0_transfun[self.stratum_lens_idx]
                # If feat0_transfun is specified, do feature transformation first.
                # If custom data indices are specified, select the subset of data first.
                # In both cases, we cannot reuse the cached stats for the feature stats and need recompute
                if feat0_transfun or self.dindices_are_stratified:
                    if self.dindices_are_stratified:
                        feat0 = self.feat0[self.lens_stratum_dindices]
                        # no need to select the stratum from feat1, 
                        # as feat1 is extracted from the stratum as above
                    else:
                        feat0 = self.feat0
                        
                    with torch.no_grad():
                        if feat0_transfun:
                            feat0 = feat0_transfun(feat0)
                        feat0_chan_pos_mean, feat0_chan_neg_mean, feat0_chan_pos_std, \
                            feat0_chan_neg_std = get_chan_pos_neg_stats(feat0)
                        
                        # Recompute topk activations & locations
                        feat0_2d, feat0_high, feat0_high_coords, feat0_low, feat0_low_coords = \
                            get_top_activations( feat0, self.feat0_act_topk )
                else:
                    feat0 = self.feat0
                    feat0_chan_pos_mean, feat0_chan_neg_mean, feat0_chan_pos_std, feat0_chan_neg_std = \
                        self.feat0_chan_pos_mean, self.feat0_chan_neg_mean, self.feat0_chan_pos_std, self.feat0_chan_neg_std
                    # Use cached topk activations & locations
                    feat0_2d, feat0_high, feat0_high_coords, feat0_low, feat0_low_coords = \
                        self.feat0_2d, self.feat0_high, self.feat0_high_coords, \
                        self.feat0_low, self.feat0_low_coords

                feat1_cbhw = feat1.permute(1, 0, 2, 3).contiguous()
                feat1_2d = feat1_cbhw.view(feat1_cbhw.shape[0], -1)
                feat1_feat0high_2d = torch.gather(feat1_2d, 1, feat0_high_coords)
                feat1_feat0high_1d = feat1_feat0high_2d.view(-1)
                feat1_feat0low_2d  = torch.gather(feat1_2d, 1, feat0_low_coords)
                feat1_feat0low_1d = feat1_feat0low_2d.view(-1)

                feat1_high, feat1_high_coords = feat1_2d.topk(self.feat1_act_topk, dim=1)
                feat0_feat1high_2d = torch.gather(feat0_2d, 1, feat1_high_coords)
                feat0_feat1high_1d = feat0_feat1high_2d.view(-1)
                feat0_high_1d = feat0_high.view(-1)
                feat1_high_1d = feat1_high.view(-1)
                feat0_low_1d  = feat0_low.view(-1)
                
                with torch.no_grad():
                    feat0_01_high = torch.cat( [ feat0_high_1d, feat0_feat1high_1d ] )
                    feat1_01_high = torch.cat( [ feat1_feat0high_1d, feat1_high_1d ] )
                    high_act_corr = pearson(feat0_01_high, feat1_01_high)
                    low_act_corr  = pearson(feat0_low,  feat1_feat0low_1d)
                    feat0_hwmean = F.relu(feat0).mean(dim=3).mean(dim=2)
                    feat1_hwmean = F.relu(feat1).mean(dim=3).mean(dim=2)
                    meanfeat_corr  = pearson(feat0_hwmean,  feat1_hwmean)
                    allfeat_corr = pearson(F.relu(feat0), F.relu(feat1))
                    self.meanfeat_corr = meanfeat_corr
                    self.allfeat_corr  = allfeat_corr
                    self.high_act_corr = high_act_corr
                    if verbose:
                        print("%s %s H: %.3f/%.3f, L: %.3f/%.3f, cor: %.3f/%.3f/%.3f/%.3f" % \
                                            (self.curr_lens_name, compfeat_phase_name,
                                             feat0_high.mean().item(),
                                             feat1_high.mean().item(),
                                             feat0_low.mean().item(),
                                             feat1_feat0low_1d.mean().item(),
                                             high_act_corr, low_act_corr,
                                             meanfeat_corr, allfeat_corr))

                self.compute_whole_activ_loss(feat0, feat1, verbose=verbose)
                self.compute_top_activ_loss(feat0_high, feat1_feat0high_2d, feat0_feat1high_2d,
                                            feat1_high, feat0_low, feat1_feat0low_2d, 
                                            feat0_chan_pos_mean, feat0_chan_neg_mean, 
                                            feat0_chan_pos_std, feat0_chan_neg_std, 
                                            verbose=verbose)


        else:
            with torch.no_grad():
                # stratum_lens_idx == 0 should never happen here. 
                # In this case, contrast_feat_low_high() is skipped
                # feat1 contains features of all images. Will select a stratum later
                if self.stratum_lens_idx == 0:
                    feat1 = self.feat0
                else:
                    # if CALC_ORIGFEAT_PEARSON, only calculate pearson correlations between 
                    # orig_feat(orig_images) <=> inv_trans( orig_feat( trans_images) ) )
                    # do inverse transformation to orig_feat only, without going through lens
                    # if using_rot_lenses, any inv_transforms[stratum_lens_idx] is not None
                    # (note stratum_lens_idx > 0 here)
                    if self.dindices_are_stratified:
                        x_stratum = get_dict_stratum(self.x, self.lens_stratum_dindices)
                    else:
                        x_stratum = self.x
                        
                    if CALC_ORIGFEAT_PEARSON and self.inv_transforms[self.stratum_lens_idx]:
                        feat1 = self.inv_transforms[self.stratum_lens_idx]( x_stratum['orig_feat'] )
                    # Otherwise, the features go through the lens. 
                    # Pearson correlations between lens-transformed features and original features 
                    # are calculated.
                    else:
                        feat1 = self.all_lenses[self.stratum_lens_idx](x_stratum)

                # feat0_transfun is a transformation function to feat0
                # By default feat0_transfun = None
                # feat0_transfun is specified among origsize_lenses 
                # (doing downsampling to original features)
                feat0_transfun = self.all_feat0_transfun[self.stratum_lens_idx]
                # If feat0_transfun is specified, do feature transformation first.
                # If custom data indices are specified, select the subset of data first.
                # In both cases, we cannot reuse the cached stats for the feature stats and need recompute
                if feat0_transfun or self.dindices_are_stratified:
                    if self.dindices_are_stratified:
                        feat0 = self.feat0[self.lens_stratum_dindices]
                        # no need to select the stratum from feat1, 
                        # as feat1 is extracted from the stratum as above
                    else:
                        feat0 = self.feat0
                        
                    if feat0_transfun:
                        feat0 = feat0_transfun(feat0)
                    feat0_chan_pos_mean, feat0_chan_neg_mean, feat0_chan_pos_std, feat0_chan_neg_std = get_chan_pos_neg_stats(feat0)
                    # recompute topk activations & locations
                    try:
                        feat0_2d, feat0_high, feat0_high_coords, feat0_low, feat0_low_coords = \
                        get_top_activations( feat0, self.feat0_act_topk )
                    except:
                        pdb.set_trace()
                else:
                    feat0 = self.feat0
                    feat0_chan_pos_mean, feat0_chan_neg_mean, feat0_chan_pos_std, feat0_chan_neg_std = \
                        self.feat0_chan_pos_mean, self.feat0_chan_neg_mean, self.feat0_chan_pos_std, self.feat0_chan_neg_std
                    # use cached topk activations & locations
                    feat0_2d, feat0_high, feat0_high_coords, feat0_low, feat0_low_coords = \
                        self.feat0_2d, self.feat0_high, self.feat0_high_coords, \
                        self.feat0_low, self.feat0_low_coords

                feat1_cbhw = feat1.permute(1, 0, 2, 3).contiguous()
                feat1_2d = feat1_cbhw.view(feat1_cbhw.shape[0], -1)
                feat1_feat0high_2d = torch.gather(feat1_2d, 1, feat0_high_coords)
                feat1_feat0high_1d = feat1_feat0high_2d.view(-1)
                feat1_feat0low_2d  = torch.gather(feat1_2d, 1, feat0_low_coords)
                feat1_feat0low_1d = feat1_feat0low_2d.view(-1)

                feat1_high, feat1_high_coords = feat1_2d.topk(self.feat1_act_topk, dim=1)
                feat0_feat1high_2d = torch.gather(feat0_2d, 1, feat1_high_coords)
                feat0_feat1high_1d = feat0_feat1high_2d.view(-1)

                feat0_high_1d = feat0_high.view(-1)
                feat1_high_1d = feat1_high.view(-1)
                feat0_low_1d  = feat0_low.view(-1)

                feat0_01_high = torch.cat( [ feat0_high_1d, feat0_feat1high_1d ] )
                feat1_01_high = torch.cat( [ feat1_feat0high_1d, feat1_high_1d.view(-1) ] )

                high_act_corr = pearson(feat0_01_high, feat1_01_high)
                low_act_corr  = pearson(feat0_low,  feat1_feat0low_1d)
                feat0_hwmean = F.relu(feat0).mean(dim=3).mean(dim=2)
                feat1_hwmean = F.relu(feat1).mean(dim=3).mean(dim=2)
                meanfeat_corr  = pearson(feat0_hwmean,  feat1_hwmean)
                allfeat_corr = pearson(F.relu(feat0), F.relu(feat1))
                self.meanfeat_corr = meanfeat_corr
                self.allfeat_corr = allfeat_corr
                self.high_act_corr = high_act_corr
                if verbose:
                    print("%s %s H: %.3f/%.3f, L: %.3f/%.3f, cor: %.3f/%.3f/%.3f/%.3f" % \
                                            (self.curr_lens_name, compfeat_phase_name,
                                             feat0_high.mean().item(),
                                             feat1_high.mean().item(),
                                             feat0_low.mean().item(),
                                             feat1_feat0low_1d.mean().item(),
                                             high_act_corr, low_act_corr,
                                             meanfeat_corr, allfeat_corr))


    def compute_whole_activ_loss(self, feat0, feat1, verbose=False):
        if USING_MAE:
            stratum_whole_activ_loss = self.maeloss(feat0, feat1) * 3
        else:
            stratum_whole_activ_loss = self.mseloss(feat0, feat1) * 3
        self.whole_activ_loss += stratum_whole_activ_loss
        self.all_lenses_whole_activ_loss[self.stratum_lens_idx] = stratum_whole_activ_loss.item()
        if verbose:
            print("%s all act loss: %.3f" %(self.curr_lens_name, stratum_whole_activ_loss))

    def compute_top_activ_loss(self, feat0_high, feat1_feat0high_2d, feat0_feat1high_2d,
                               feat1_high, feat0_low, feat1_feat0low_2d, 
                               feat0_chan_pos_mean, feat0_chan_neg_mean, 
                               feat0_chan_pos_std, feat0_chan_neg_std, 
                               verbose=False):
        # If using amp, feat* are float16. Cast to float32 to avoid overflow when doing summation
        # Only optimize w.r.t. feat1_* (yielded by lenses), 
        # So detach all feat0_* tensors to avoid accidentally involving them in the computation graph
        feat0_high, feat1_feat0high_2d, \
        feat0_feat1high_2d, feat1_high, \
        feat0_low, feat1_feat0low_2d = \
            feat0_high.detach().float(), feat1_feat0high_2d.float(), \
            feat0_feat1high_2d.detach().float(), feat1_high.float(), \
            feat0_low.detach().float(), feat1_feat0low_2d.float()
            
        feat0_high_diff = feat1_feat0high_2d - feat0_high
        feat1_high_diff = feat1_high - feat0_feat1high_2d
        feat0_low_diff =  feat1_feat0low_2d - feat0_low
        # Set low_rescale = 1.0 to disable
        low_rescale = feat0_chan_pos_std.sum() / feat0_chan_neg_std.sum()
        avg_feat0_chan_pos_std = feat0_chan_pos_std.mean()
        avg_feat0_chan_neg_std = feat0_chan_neg_std.mean()
        feat0_chan_pos_std = torch.clamp(feat0_chan_pos_std, avg_feat0_chan_pos_std / 2)
        feat0_chan_neg_std = torch.clamp(feat0_chan_neg_std, avg_feat0_chan_neg_std / 2)
        
        # low/high_underdiff: undershoot (too small when positive, or too big when negative)
        # low/high_overdiff:  overshoot  (too big when positive, or too small when negative)
        # We don't care about feat0_high_overdiff or feat1_high_underdiff. 
        # Because feat1 is allowed to overshoot  feat0 on feat0 top activations (with slight penalty),
        # and     feat1 is allowed to undershoot feat0 on feat1 top activations.
        feat0_high_underdiff_mask = (feat0_high_diff < 0).float()
        feat1_high_overdiff_mask  = (feat1_high_diff > 0).float()
        feat0_low_underdiff_mask  = (feat0_low_diff  > 0).float()
        feat0_low_overdiff_mask   = (feat0_low_diff  < 0).float()
        
        # Negative where feat1 < feat0, at top positive activations of feat0. 0 otherwise
        feat0_high_underdiff = feat0_high_diff * feat0_high_underdiff_mask
        # Positive where feat1 > feat0, at top positive activations of feat1. 0 otherwise
        feat1_high_overdiff  = feat1_high_diff * feat1_high_overdiff_mask
        # Positive where feat1 > feat0, at top negative activations of feat0. 0 otherwise
        feat0_low_underdiff  = feat0_low_diff  * feat0_low_underdiff_mask
        # Negative where feat1 < feat0, at top negative activations of feat0. 0 otherwise
        feat0_low_overdiff   = feat0_low_diff  * feat0_low_overdiff_mask

        if self.top_act_weighted_by_feat0:
            activ_undershoot_loss =  -(feat0_high_underdiff * feat0_high).sum() \
                                      / feat0_high_underdiff_mask.sum() \
                                     +(feat0_low_underdiff  * -feat0_low).sum() * low_rescale \
                                      / feat0_low_underdiff_mask.sum()
        else:
            activ_undershoot_loss =  -feat0_high_underdiff.sum() \
                                      / feat0_high_underdiff_mask.sum() \
                                     +feat0_low_underdiff.sum() * low_rescale \
                                      / feat0_low_underdiff_mask.sum()
                                                                            
        # To simplify the procedure, overshoot loss is not weighted by feature values or channel STD
        activ_overshoot_loss  =   feat1_high_overdiff.mean() \
                                 -feat0_low_overdiff.mean() * low_rescale

        if DEBUG:
            pdb.set_trace()

        overshoot_discount = self.overshoot_discount # 0.2
        stratum_top_activ_loss = activ_undershoot_loss + activ_overshoot_loss * overshoot_discount
        stratum_top_activ_loss = stratum_top_activ_loss * 3

        avg_feat0_high_underdiff = ( feat0_high_underdiff.sum() / (feat0_high_underdiff_mask.sum() + 0.001) ).item()
        avg_feat1_high_overdiff  = ( feat1_high_overdiff.sum()  / (feat1_high_overdiff_mask.sum() + 0.001) ).item()
        avg_feat0_low_overdiff   = ( feat0_low_overdiff.sum()   / (feat0_low_overdiff_mask.sum() + 0.001) ).item()
        avg_feat0_low_underdiff  = ( feat0_low_underdiff.sum()  / (feat0_low_underdiff_mask.sum() + 0.001) ).item()

        self.top_activ_loss += stratum_top_activ_loss
        self.all_lenses_top_activ_loss[self.stratum_lens_idx] = stratum_top_activ_loss.item()

        if verbose:
            # A group of typical stats. No need to print every iteration
            # feat0 pos mean/std 0.835/0.931, neg mean/std -0.381/0.334
            #print("feat0 pos mean/std %.3f/%.3f, neg mean/std %.3f/%.3f" % \
            #        (feat0_chan_pos_mean, feat0_chan_pos_std, feat0_chan_neg_mean, feat0_chan_neg_std))
                    
            print("%s lowres %.2f, h> %.3f, h< %.3f, l> %.3f, l< %.3f" % \
                    (self.curr_lens_name, low_rescale, avg_feat1_high_overdiff, 
                    avg_feat0_high_underdiff, avg_feat0_low_overdiff, 
                    avg_feat0_low_underdiff))

            print("%s undershoot loss: %.2f, overshoot loss: %.2f" % \
                    (self.curr_lens_name, activ_undershoot_loss.item(),
                     activ_overshoot_loss.item() * overshoot_discount))
