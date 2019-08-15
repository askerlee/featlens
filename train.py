from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.nn import Parameter
import torchvision.transforms as transforms
from imagenet_seq.data import Loader
from dotmap import DotMap
import os
import pdb
import time
import tensorpack.dataflow as df
from tensorpack import imgaug
import cv2
import numpy as np
from PIL import Image
import math
import argparse
import re
from featlens import FeatLens, Flatten
from resnet import resnet101, resnet50, resnet34, resnet18
from torchvision.models import densenet, vgg16, vgg16_bn
                    
from imagenet1000_clsid_to_human import id2label

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='1-layer Hebbian CNN')

    parser.add_argument('--net', dest='cnn_type',
                      help='type of the pretrained network',
                      default='resnet101', type=str)
    parser.add_argument('--cp', dest='cp_filepath',
                        help='model checkpoint to load',
                        default=None, type=str)
    parser.add_argument('--lr', dest='lr',
                        help='initial learning rate for supervised learning',
                        default=0.2, type=float)
    parser.add_argument('--layer', dest='featlens_layers_str',
                        help='resnet layer applied with FeatLens',
                        default='4', type=str)
    parser.add_argument('--2', dest='two_layer_conv',
                      help='Use two layers of conv',
                      action='store_true')
                        
    parser.add_argument('--b', dest='train_batch_size',
                        help='Training batch size',
                        default=120, type=int)
    parser.add_argument('--numsec', dest='num_sectors',
                        help='Number of sectors of the circle (possible angles of rotations)',
                        default=4, type=int)
    parser.add_argument('--context', dest='context_size',
                        help='Size of feature map context',
                        default=0, type=int)
    parser.add_argument('--groundrot', dest='out_groudrot_featlens',
                      help='Disable rotation-equivariance group convolution',
                      action='store_true')
    parser.add_argument('--updateallbn', dest='update_all_batchnorms',
                      help='Update all pretrained (default: only update FeatLens batchnorms)',
                      action='store_true')
    parser.add_argument('--adam', dest='use_sgd',
                      help='Use adam optimizer',
                      action='store_false')
    parser.add_argument('--eval', dest='eval_only',
                      help='Evaluate trained model on validation data',
                      action='store_true')
    parser.add_argument('--bnonly', dest='bn_only',
                      help='FeatLens only does orientation-specific bn',
                      action='store_true')
    parser.add_argument('--disable', dest='enable_featlens',
                      help='Disable featlens',
                      action='store_false')
                        
    # verbosity of debug info
    parser.add_argument('--d', dest='debug',
                      help='debug mode',
                      action='store_true')
    args = parser.parse_args()
    return args

args_dict = {  'max_epochs': 1, 
               'start_epoch': 0,
               'num_workers': 4,
               'save_model_interval': 100,
               'print_cls_stats_iters': 500,
               'disp_interval': 10,
               'num_mix_comps': 5,
               'featlens_do_equalvote': True,
               # topk of each feature map (should be multiplied with batch size)
               'exc_topk': 5, 
               'calc_all_exc_loss': False,
               'use_softmax': True,
               'builtin_bn': True,
               'update_all_batchnorms': False,
               'lr_decay_stepsize': 200,
               'add_residual': True,
               'do_vote': False,
            }
            
cmd_args = parse_args()
args_dict.update( vars(cmd_args) )
args = DotMap(args_dict)
if args.two_layer_conv:
    args.train_batch_size = 300
    
args.test_batch_size = int(args.train_batch_size * 1.6)

if not args.out_groudrot_featlens:
    args.do_vote = True
    
if not args.use_softmax:
    args.lr /= 50

if args.adam:
    args.lr /= 10

cnntype_match = re.match("^(resnet|vgg)", args.cnn_type)        
if not cnntype_match:
    raise NotImplementedError
cnn_cat = cnntype_match.group(1)
    
def nparray2tensor(x, cuda, rot_angle):
    normalize = transforms.Compose(
                    [ 
                      transforms.RandomRotation((rot_angle, rot_angle), 
                                                resample=Image.BILINEAR, expand=True),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                    ] )

    for i, xi in enumerate(x):
        xi_img = Image.fromarray(xi)
        xi_tensor = normalize(xi_img)
        
        if i == 0:
            x2_shape = [x.shape[0]] + list(xi_tensor.shape)
            if cuda:
                x2 = torch.zeros(x2_shape).cuda()
            else:
                x2 = torch.zeros(x2_shape)
        # although normalize is done in-place,
        # it's on a temporary tensor whose ref is returned
        x2[i] = xi_tensor  
          
    return x2

if args.cnn_type == 'resnet101':
    net = resnet101(pretrained=True)
elif args.cnn_type == 'resnet50':
    net = resnet50(pretrained=True)
elif args.cnn_type == 'resnet34':
    net = resnet34(pretrained=True)
elif args.cnn_type == 'resnet18':
    net = resnet18(pretrained=True)
elif args.cnn_type == 'vgg':
    net = vgg16_bn(pretrained=True)
else:
    raise NotImplementedError
    
if args.update_all_batchnorms:
    net.train()
else:
    net.eval()
    
net.cuda()

orig_convs = []
orig_bns = []
orig_blocks = []
featlenses = []
args.enable_featlens = True
# if args.enable_featlens = False, the original pretrained model will be evaluated
# should yield the same performance as when featlens.set_use_featlens(False)
if args.eval_only and args.cp_filepath is None:
    args.enable_featlens = False
all_featlens_params = []
featlens_layer_indices = args.featlens_layers_str.split(',')
featlens_layer_indices = [ int(layer) for layer in featlens_layer_indices ]
featlens_names = []
small_resnet_num_chans  = [0, 64,  128, 256, 512]
big_resnet_num_chans    = [0, 256, 512, 1024, 2048]
resnet_overshoot_discounts = [-1, -1, -1,  0.5,  0.2]
overshoot_discounts = []
focal_excite_featlenses = []
add_residual_flags = []
excite_loss_weights = []

if args.enable_featlens:  
    for i, featlens_layer_idx in enumerate(featlens_layer_indices):
        if cnn_cat == 'resnet':
            if featlens_layer_idx == 3:
                featlens_layer = net.layer3
            elif featlens_layer_idx == 4:
                featlens_layer = net.layer4
            else:
                raise NotImplementedError
                
            if args.cnn_type == 'resnet34' or args.cnn_type == 'resnet18':
                end2_block = featlens_layer[-1]
                orig_blocks.append(end2_block)
                # creating a conv that transforms the residual features
                # orig_convs.append(small_resnet_num_chans[featlens_layer_idx])
                # orig_bns.append(None)
                orig_convs.append(end2_block.conv2)
                orig_bns.append(end2_block.bn2)
                # original bn will be built-in in FeatLens
                # end2_block.bn2 = nn.Sequential()
                featlens_names += [ "l%dconv2" %(featlens_layer_idx) ]
            else:
                end2_block = featlens_layer[-1]
                orig_blocks.append(end2_block)
                # creating a conv that transforms the residual features
                # orig_convs.append(big_resnet_num_chans[featlens_layer_idx])
                # orig_bns.append(None)
                orig_convs.append(end2_block.conv3)
                orig_bns.append(end2_block.bn3)
                # original bn will be built-in in FeatLens
                # end2_block.bn3 = nn.Sequential()
                featlens_names += [ "l%dconv3" %(featlens_layer_idx) ]
                
            overshoot_discounts += [ resnet_overshoot_discounts[featlens_layer_idx] ]
            add_residual_flags += [ args.add_residual ]
            excite_loss_weights += [ 2**i ]
                   
        elif cnn_cat == 'vgg':
            if featlens_layer_idx == 3:
                orig_convs.append(net.features[30])
                orig_bns.append(net.features[31])
                # disable the following bn (original bn will be built-in in FeatLens)
                net.features[31] = nn.Sequential()
            elif featlens_layer_idx == 4:
                orig_convs.append(net.features[40])
                orig_bns.append(net.features[41])
                # disable the following bn (original bn will be built-in in FeatLens)
                net.features[41] = nn.Sequential()
            else:
                raise NotImplementedError
            featlens_names += [ "l%dconv1" %(featlens_layer_idx) ]
            overshoot_discounts += [ 0.2 ]
            add_residual_flags += [ False ]
            excite_loss_weights += [ 2**i ]
        else:
            raise NotImplementedError

    for i, orig_conv in enumerate(orig_convs):
        orig_bn = orig_bns[i]
        featlens = FeatLens( featlens_names[i], 
                                    orig_conv, orig_bn,
                                    builtin_bn=args.builtin_bn,
                                    num_sectors=args.num_sectors, 
                                    num_mix_comps=args.num_mix_comps, 
                                    featlens_do_equalvote=args.featlens_do_equalvote,
                                    exc_topk=args.exc_topk * args.train_batch_size, 
                                    use_softmax=args.use_softmax, 
                                    context_size=args.context_size, 
                                    bn_only=args.bn_only,
                                    overshoot_discount=overshoot_discounts[i],
                                    add_residual=add_residual_flags[i],
                                    two_layer_conv=args.two_layer_conv,
                                    do_vote=args.do_vote,
                                    is_debug=args.debug,
                                    calc_all_exc_loss=args.calc_all_exc_loss )
                            
        featlens.cuda()
        featlens.train()
        # fix orig_bn (part of orig_conv now) for fair evaluation
        if orig_bn:
            featlens.orig_conv[1].eval()
            
        featlenses.append(featlens)
        featlens.set_update_featlens(False)
        featlens.set_out_groudrot_featlens(args.out_groudrot_featlens)
        all_featlens_params += list(featlens.parameters())
    
    for i, featlens_layer_idx in enumerate(featlens_layer_indices):
        if cnn_cat == 'resnet':
            if args.cnn_type == 'resnet34' or args.cnn_type == 'resnet18':
                orig_blocks[i].conv2 = featlenses[i]
            else:
                orig_blocks[i].conv3 = featlenses[i]
        
        elif cnn_cat == 'vgg':
            if featlens_layer_idx == 3:
                net.features[30] = featlenses[i]
            elif featlens_layer_idx == 4:
                net.features[40] = featlenses[i]

    '''
    if cnn_cat == 'resnet':
        focal_excite_featlenses = featlenses[-2:]
    elif cnn_cat == 'vgg':
        focal_excite_featlenses = featlenses[-1:]
    '''
    
    excite_loss_weights = np.array(excite_loss_weights, dtype=float)
    excite_loss_weights /= excite_loss_weights[-1]
    
num_featlenses = len(featlenses)

if args.cp_filepath is not None:
    match = re.search('([0-9a-z]+)-sl([0-9,]+)', args.cp_filepath)
    if match is not None:
        # last_unsup_layer is always extracted from the checkpoint filename
        cnn_type = match.group(1)
        featlens_layers_str = match.group(2)
        if args.featlens_layers_str != featlens_layers_str:
            print("argument --layer '%s' != cp '%s'" %(args.featlens_layers_str, featlens_layers_str))
            exit(0)
    else:
        pdb.set_trace()

    state_dict = torch.load(args.cp_filepath)
    net.load_state_dict(state_dict)

train_loader = Loader('train', batch_size=args.train_batch_size, shuffle=False, 
                        num_workers=args.num_workers, out_tensor=False)
test_loader  = Loader('val',   batch_size=args.test_batch_size, shuffle=False, 
                        num_workers=args.num_workers, out_tensor=False)
crossEnt = nn.CrossEntropyLoss()
# placeholder
im_data = torch.cuda.FloatTensor(1)
stage_names = ['Train', 'Test ']

if args.eval_only:
    args.max_epochs = 1
    data_loader = test_loader
else:
    data_loader = train_loader
    
if args.enable_featlens:
    sgd_optimizer    = torch.optim.SGD(all_featlens_params, lr=args.lr, momentum=0.9)
    adam_optimizer   = torch.optim.Adam(all_featlens_params, lr=args.lr)
else:
    dummy_param = [ torch.FloatTensor(1).cuda() ]
    sgd_optimizer    = torch.optim.SGD(dummy_param, lr=args.lr, momentum=0.9)
    adam_optimizer   = torch.optim.Adam(dummy_param, lr=args.lr)
    
exp_lr_scheduler = StepLR(sgd_optimizer, step_size=args.lr_decay_stepsize, gamma=0.5)

if args.use_sgd:
    optimizer = sgd_optimizer
else:
    optimizer = adam_optimizer

for epoch in range(args.start_epoch, args.max_epochs):
    start = time.time()
    print("Epoch %d" %epoch)
    data_iter = iter(data_loader)
    iters_per_epoch = len(data_loader)
    disp_correct_insts_t1 = np.zeros((2, args.num_sectors))
    disp_correct_insts_t5 = np.zeros((2, args.num_sectors))
    all_correct_insts_t1 = np.zeros((2, args.num_sectors))
    all_correct_insts_t5 = np.zeros((2, args.num_sectors))
    
    correct_insts_cls = torch.zeros(2, args.num_sectors, 1000).cuda()
    disp_total_insts_cls = torch.zeros(2, args.num_sectors, 1000).cuda()
    disp_total_insts = np.zeros((2, args.num_sectors))
    total_insts = np.zeros((2, args.num_sectors))
    
    disp_cls_loss = np.zeros((2, args.num_sectors))
    disp_excite_loss = np.zeros((2, num_featlenses, args.num_sectors))
    upright_count = np.zeros((2, num_featlenses, args.num_sectors))
    stage = 0
    
    for step in range(iters_per_epoch):
        batch_data = next(data_iter)

        for sec_idx in range(args.num_sectors):
            rot_angle = sec_idx * 360.0 / args.num_sectors
            im_data0 = nparray2tensor(batch_data[0], cuda=True, rot_angle=rot_angle)
            batch_labels = torch.LongTensor(batch_data[1]).cuda()

            im_data.resize_(im_data0.shape).copy_(im_data0)
            for featlens in featlenses:
                featlens.sec_idx = sec_idx
                featlens.num_iter = step
                
            for stage in range(1, -1, -1):
                if stage == 0:  
                    for featlens in featlenses:
                        featlens.set_use_featlens(False)
                        if not args.eval_only:
                            featlens.set_update_featlens(True)
                else:
                    for featlens in featlenses:
                        featlens.set_use_featlens(True)
                        featlens.set_update_featlens(False)
                                    
                with torch.no_grad():
                    cls_scores = net(im_data)

                if stage == 0 and sec_idx != 0 and not args.eval_only:
                    total_excite_loss = 0
                    for i, featlens in enumerate(featlenses):
                        loss_weight = excite_loss_weights[i]
                        total_excite_loss += loss_weight * featlens.excite_loss
                    
                    optimizer.zero_grad()
                    total_excite_loss.backward()
                    optimizer.step()
                    
                    for i, featlens in enumerate(featlenses):
                        featlens.compute_feat_low_high('new', do_compute_loss=False)
                        
                cls_probs = F.softmax(cls_scores, dim=-1)
                cls_loss = crossEnt(cls_probs, batch_labels)

                # sum of cls_loss over a disp_interval. 
                # for computing the average loss during this interval
                disp_cls_loss[stage, sec_idx] += cls_loss.item()
                if args.enable_featlens:
                    for i, featlens in enumerate(featlenses):
                        disp_excite_loss[stage, i, sec_idx] += featlens.excite_loss.item()
                        upright_count[stage, i, sec_idx] += featlens.upright_count
                
                pred_labels_t1 = cls_probs.argmax(dim=1)
                pred_labels_t5 = cls_probs.topk(5, dim=1)[1]
                disp_correct_insts_t1[stage, sec_idx] += ( pred_labels_t1 == batch_labels ).sum().item()
                is_t5_correct = ( pred_labels_t5 == batch_labels.view(-1, 1) ).sum(dim=1)
                disp_correct_insts_t5[stage, sec_idx] += is_t5_correct.sum().item()
                disp_total_insts[stage, sec_idx] += len(batch_labels)
                
                if is_t5_correct.sum() > 0:
                    batch_correct_cls = torch.bincount( batch_labels[is_t5_correct.byte()], minlength=1000 )
                    correct_insts_cls[stage, sec_idx] += batch_correct_cls.float()
                    
                batch_total_insts_cls = torch.bincount( batch_labels, minlength=1000 )
                disp_total_insts_cls[stage, sec_idx] += batch_total_insts_cls.float()

        disp_stats = reset_train_stats = reset_test_stats = False
        if (step+1) % args.disp_interval == 0:
            end = time.time()
            disp_cls_loss    /= args.disp_interval
            disp_excite_loss /= args.disp_interval
            upright_count        /= args.disp_interval
            disp_correct_perc_t1 = disp_correct_insts_t1 / disp_total_insts
            disp_correct_perc_t5 = disp_correct_insts_t5 / disp_total_insts
            
            for stage in range(2):
                for sec_idx in range(args.num_sectors):
                    rot_angle = sec_idx * 360.0 / args.num_sectors
                    excite_loss_str   = '/'.join( ["%.2f" %(loss) for loss in disp_excite_loss[stage, :, sec_idx] ])
                    upright_count_str = '/'.join( ["%d"  %(count) for count in upright_count[stage, :, sec_idx] ])
                    print("%3d/%3d %s rot %03d cl: %.2f, acc: %.3f/%.3f" \
                            % (step+1, iters_per_epoch, stage_names[stage], 
                               rot_angle, disp_cls_loss[stage, sec_idx],
                               disp_correct_perc_t1[stage, sec_idx], 
                               disp_correct_perc_t5[stage, sec_idx]))

                    print("el: %s, up: %s" %(excite_loss_str, upright_count_str))

            avg_perc_diff_t1 = ( disp_correct_perc_t1[1, 1:] - disp_correct_perc_t1[0, 1:] ).mean()
            avg_perc_diff_t5 = ( disp_correct_perc_t5[1, 1:] - disp_correct_perc_t5[0, 1:] ).mean()
            print("%3d t1 diff: %.1f%%, t5 diff: %.1f%%" %(step+1, avg_perc_diff_t1 * 100, avg_perc_diff_t5 * 100))
            
            disp_cls_loss[...] = 0
            disp_excite_loss[...] = 0
            upright_count[...] = 0
            
            all_correct_insts_t1 += disp_correct_insts_t1
            all_correct_insts_t5 += disp_correct_insts_t5
            total_insts += disp_total_insts

            avg_perc_diff_t1 = ( all_correct_insts_t1[1] - all_correct_insts_t1[0] ) / total_insts[0]
            avg_perc_diff_t5 = ( all_correct_insts_t5[1] - all_correct_insts_t5[0] ) / total_insts[0]
            print("e%d  t1 diff: %.1f%%, t5 diff: %.1f%%" %(epoch, avg_perc_diff_t1[1:].mean() * 100, 
                                                                avg_perc_diff_t5[1:].mean() * 100))

            disp_correct_insts_t1[...] = 0
            disp_correct_insts_t5[...] = 0
            disp_total_insts[...] = 0
            start = time.time()
    
        '''
        if (step+1) % args.print_cls_stats_iters == 0:
            total_insts_cls2 = total_insts_cls.clone()
            total_insts_cls2[total_insts_cls2==0] = 1
            acc_cls = correct_insts_cls / total_insts_cls2
            
            for stage in range(2):
                for sec_idx in range(args.num_sectors):
                    rot_angle = sec_idx * 360.0 / args.num_sectors
                    topk = 10
                    top_acc, top_cls = acc_cls[stage, sec_idx].topk(topk)
                    print("%s rot %03d Top %d performing classes:" %(stage_names[stage], rot_angle, topk))
                    for i in range(topk):
                        cls_id = top_cls[i].item()
                        print("%d-%s: %.3f %d/%d" %(cls_id, id2label[cls_id], top_acc[i],
                                                correct_insts_cls[stage, sec_idx, cls_id], 
                                                total_insts_cls[stage, sec_idx, cls_id]))
            print()    
            correct_insts_cls.zero_()
            total_insts_cls.zero_()
            '''

        if (not args.eval_only) and (step+1) % args.save_model_interval == 0:
            if not os.path.isdir('models'):
                os.makedirs('models')
            cp_filepath = "models/%s-sl%s-%04d.pth" %(args.cnn_type, args.featlens_layers_str, step+1)
            torch.save(net.state_dict(), cp_filepath)
            print("Save checkpoint '%s'" %cp_filepath)
        
        # if use adam, this scheduler has no effect
        exp_lr_scheduler.step()
        