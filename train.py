from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler
import torch.backends.cudnn as cudnn
import pdb
from torch.nn import Parameter
from torch._six import inf
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from lmdbloader import LMDBLoader
from easydict import EasyDict as edict
import os
import time
import datetime
import tensorpack.dataflow as df
from tensorpack import imgaug
import cv2
import numpy as np
from PIL import Image
import math
import argparse
import re
import featlens
from featlens import LensKit, Flatten, RotateTensor4d, ScaleTensor4d, \
                     AverageMeter, AverageMeters, print_maxweight, identity

from resnet import resnet101, resnet50, resnet34, resnet18
from torchvision.models import densenet, vgg16, vgg16_bn
from ranger import Ranger
import inspect
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

# some fixed hyperparameters
args_dict = {  'num_workers': 4,
               'adam_epsilon': 1e-8,
               'builtin_bn': True,
               #'max_grad_value': 10.,
            }

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Trainer for featlens on resnet')

    parser.add_argument('--ep', dest='total_epochs',
                        help='total number of epochs',
                        default=-1, type=int)
    parser.add_argument('--arch', dest='architecture',
                      help='type of the pretrained network',
                      default='resnet101', type=str)
    parser.add_argument('--cp', dest='lenses_cp_filepath',
                        help='lenses checkpoint to load',
                        default=None, type=str)
    parser.add_argument('--hostcp', dest='host_cp_filepath',
                        help='host checkpoint to load',
                        default=None, type=str)
    parser.add_argument('--redo-cp-iters', dest='skip_cp_iters',
                        help='Skip iterations that the checkpoint has been trained on',
                        action='store_false')

    parser.add_argument('--xlayer', dest='using_xlayer_resnet',
                        help='Use pretrained resnet (frozen) + an extra trainable layer',
                        action='store_true')
    # Default not to do scaling together with rotation. 
    # There's some implementation nuisance to do so.
    parser.add_argument('--doscale', dest='using_scaling_lenses',
                        help='Evaluate scaling lenses',
                        action='store_true')
    # Default to do rotation. Can be suppressed by --norot.
    parser.add_argument('--norot', dest='using_rot_lenses',
                        help='Do not train/evaluate rotation lenses',
                        action='store_false')

    parser.add_argument('--lr', dest='lr',
                        help='initial learning rate',
                        default=-1, type=float)
    parser.add_argument('--decay', dest='weight_decay',
                        help='rate of weight decay',
                        default=-1, type=float)
    parser.add_argument('--gamma', default=-1, type=float,
                        metavar='Y', help='learning rate decay factor')
    parser.add_argument('--bs', dest='train_batch_size',
                        help='Training batch size',
                        default=256, type=int)
    parser.add_argument('--opt', dest='optimizer',
                      help='The type of optimizer',
                      default='sgd', type=str)
    parser.add_argument('--eval', dest='eval_only',
                      help='Evaluate trained model on validation data',
                      action='store_true')
    parser.add_argument('--dataset', dest='dataset',
                      help='The type of dataset (default: imagenet)',
                      default='imagenet', type=str,
                      choices=['imagenet', 'mnist', 'cifar10', 'open'])
    parser.add_argument('--guessrot', dest='guessing_rot_lenses',
                      help='Guess rotations (invalid when args.using_rot_lenses == False)',
                      action='store_true')

    parser.add_argument('--stage', dest='host_stage_name',
                        help='resnet stage applied with LensKit',
                        default='res4', type=str)
    parser.add_argument('--mix', dest='num_mix_comps',
                        help='Number of mixture conv components',
                        default=5, type=int)
    parser.add_argument('--context', dest='context_size',
                        help='Size of feature map context',
                        default=0, type=int)
    parser.add_argument('--updateallbn', dest='update_all_batchnorms',
                      help='Update all pretrained (default: only update LensKit batchnorms)',
                      action='store_true')
    parser.add_argument('--notopactw0', dest='top_act_weighted_by_feat0',
                      help='Do not weight top activations by feat0 values',
                      action='store_false')
    # Usually whole activation loss hurts
    parser.add_argument('--wholeact', dest='whole_act_loss_weight',
                      help='Weight of the whole activation loss',
                      type=float, default=0.)
    # Usually whole activation loss hurts
    parser.add_argument('--mae', dest='using_mae_for_whole_act_loss',
                      help='Use MAE instead of the default MSE to compute the whole activation loss',
                      action='store_true')
                      
    parser.add_argument('--nosimres', dest='sim_residual',
                      help='Do not simulate residual connections. Will halve the number of params',
                      action='store_false')
    
    parser.add_argument('--bnonly', dest='bn_only',
                      help='LensKit only does orientation-specific bn',
                      action='store_true')
    parser.add_argument('--nolens', dest='enable_lenses',
                      help='Disable featlens',
                      action='store_false')
    parser.add_argument('--aug', dest='do_aug',
                      help='Do input image augmentation during training',
                      action='store_true')

    parser.add_argument('--traintest', dest='train_on_test',
                      help='Do training on test set to evaluate domain difference',
                      action='store_true')
    parser.add_argument('--instnorm', dest='doing_instance_norm',
                      help='Do instance normalization (Do not use cached batchnorm stats)',
                      action='store_true')
    # mid_layer_chan (mid_channels) is used for scaling lenses only
    # Actual mid layer channels are mid_layer_chan * K (num_mix_comps)
    parser.add_argument('--mid', dest='mid_layer_chan',
                        help='Number of middle layer channels',
                        default=2048, type=int)
    parser.add_argument('--noup', dest='do_upsampling',
                      help='Do not upsample feature maps in scaling lenses',
                      action='store_false')
    parser.add_argument('--no-interp-origfeat', dest='interp_origfeat',
                      help='Do not interpolate original feature maps in scaling lenses',
                      action='store_false')
    parser.add_argument('--origpearson', dest='calc_origfeat_pearson',
                      help='Calculate pearson correlation',
                      action='store_true')

    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')

    parser.add_argument('--opt-level', dest='opt_level', default='O2', type=str)
    parser.add_argument('--keep-batchnorm-fp32', dest='keep_batchnorm_fp32', type=str, default=None)
    parser.add_argument('--loss-scale', dest='loss_scale', type=str, default=None)
    parser.add_argument('--no-async-prefetcher', dest='async_prefetcher', action='store_false')
    parser.add_argument('--noamp', dest='amp', help='Do not use mixed precision',
                        action='store_false')

    # verbosity of debug info
    parser.add_argument('--debug', dest='debug',
                      help='debug mode',
                      action='store_true')
    parser.add_argument("--disp", dest='disp_interval', default=10, type=int)
                      
    args = parser.parse_args()
    return args

args = edict(args_dict)
cmd_args = parse_args()
args.update( vars(cmd_args) )

# just a placeholder. Not used to initialize LensKit. 
# But 'identity' is used as img_trans() to do input image transformation. 
# So it couldn't be None.
orig_lens_spec = ['orig', identity(), None, None, 1 ],

#    lens_name,       img_trans,       feat0_transfun,     inv_trans,    topk_discount
rot_lenses_spec = [
    ['rot90',       RotateTensor4d(90),       None,   RotateTensor4d(270),   1 ],
    ['rot180',      RotateTensor4d(180),      None,   RotateTensor4d(180),   1 ],
    ['rot270',      RotateTensor4d(270),      None,   RotateTensor4d(90),    1 ],
]

# upsampling features of downsampled images to match with original features, 
# default do_upsampling=True, i.e., use scaling_upsample_lenses_spec instead of scaling_origsize_lenses_spec.
# scaling_upsample_lenses_spec performs slightly better than scaling_origsize_lenses_spec, 
# which is to downsample original features to match with lens-transformed features of downsampled images
scaling_upsample_lenses_spec = [
 # name, input image transformation,
 # transformed images: 112x112, feat: 4x4
['sc0.5,0.5',   ScaleTensor4d(0.5, 0.5),
  # feat0trans, [ TransConv params ], topk_discount (consider fewer topk activations)
  # feat0trans: preprocessing feat0 so as to do alignment, compute loss and optimize weights.
  # Here no feat0 transformation, but upsample features of transformed (scaled-down) images
  # TransConv params in MixConv2L (kernel_size, stride, padding), interp_origfeat: True/False
  None, [args.mid_layer_chan, (3,3),(2,2),(1,1), True], 0.8 ],
  # transformed images: 74x74,   feat: 3x3
['sc0.33,0.33', ScaleTensor4d(0.33, 0.33),
  None, [args.mid_layer_chan, (3,3),(2,2),(0,0), True], 0.6 ],
]

# lenses to downsample original feature maps to match feature maps on downsampled input
scaling_origsize_lenses_spec = [
 # name, input image transformation,
 # transformed images: 112x112, feat: 4x4
['sc0.5,0.5',   ScaleTensor4d(0.5, 0.5),
  # feat0trans, [ TransConv params ], topk_discount (consider fewer topk activations)
  # feat0trans: preprocessing feat0 so as to do alignment, compute loss and optimize weights.
  # Here downsample feat0, and does not change the size of features
  # extracted from transformed (scaled-down) images.
  # TransConv params in MixConv2L (kernel_size, stride, padding), interp_origfeat: True/False
  ScaleTensor4d(0.57, 0.57), [args.mid_layer_chan, (1,1),(1,1),(0,0), True], 0.8 ],
  # transformed images: 74x74,   feat: 3x3
['sc0.33,0.33', ScaleTensor4d(0.33, 0.33),
  ScaleTensor4d(0.43, 0.43), [args.mid_layer_chan, (1,1),(1,1),(0,0), True], 0.6 ],
]

def print0(*print_args, **kwargs):
    if args.local_rank == 0:
        print(*print_args, **kwargs)

def create_lenskit(net, arch_type, host_stage_name, lenses_spec, resnet_overshoot_discounts):
    # small_resnet_num_chans  = [0, 64,  128, 256, 512]
    # big_resnet_num_chans    = [0, 256, 512, 1024, 2048]

    if arch_type == 'resnet':
        if host_stage_name == 'res3':
            lenskit_layer = net.layer3
            host_layer_idx = 3
        elif host_stage_name == 'res4':
            lenskit_layer = net.layer4
            host_layer_idx = 4
        else:
            raise NotImplementedError

        if args.architecture == 'resnet34' or args.architecture == 'resnet18':
            end2_block = lenskit_layer[-1]
            orig_block = end2_block
            orig_conv = end2_block.conv2
            orig_bn = end2_block.bn2
        else:
            end2_block = lenskit_layer[-1]
            orig_block = end2_block
            orig_conv = end2_block.conv3
            orig_bn = end2_block.bn3

        overshoot_discount = resnet_overshoot_discounts[host_layer_idx]
        kit_activ_loss_weight = 1
    else:
        raise NotImplementedError

    lenskit = LensKit( host_stage_name, orig_conv, orig_bn,
                       lenses_spec = lenses_spec,
                       builtin_bn = args.builtin_bn,
                       num_mix_comps = args.num_mix_comps,
                       base_act_topk = args.base_act_topk,
                       top_act_weighted_by_feat0 = args.top_act_weighted_by_feat0,
                       overshoot_discount = overshoot_discount,
                       using_softmax_attn = True,
                       context_size = args.context_size,
                       bn_only = args.bn_only,
                       sim_residual = args.sim_residual,
                       doing_post_relu = True,
                       using_rot_lenses = args.using_rot_lenses,
                       guessing_rot_lenses = args.guessing_rot_lenses,
                       rot_cls_avgpool_size = args.rot_cls_avgpool_size,
                       num_rot_lenses = 4, # 3 rotation lenses plus the original lens
                       rot_feat_shape = args.rot_feat_shape,
                       rot_idx_mapper = None )

    net.add_module('lenskit', lenskit)

def create_imagenet_data_loader():
    normalizer = transforms.Compose(
                    [
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                    ] )

    if not args.distributed:
        train_loader = LMDBLoader('train', do_aug=args.do_aug,
                                batch_size=args.train_batch_size, shuffle=False,
                                num_workers=args.num_workers, cuda=False,
                                out_tensor=True, data_transforms=normalizer)
        val_loader   = LMDBLoader('val',   do_aug=False,
                                batch_size=args.test_batch_size, shuffle=False,
                                num_workers=args.num_workers, cuda=False,
                                out_tensor=True, data_transforms=normalizer)
    else:
        # args.num_workers = 4
        train_loader = LMDBLoader('train-%d' %args.local_rank, do_aug=args.do_aug,
                                batch_size=args.train_batch_size, shuffle=False,
                                num_workers=args.num_workers, cuda=False,
                                out_tensor=True, data_transforms=normalizer)
        val_loader   = LMDBLoader('val-%d' %args.local_rank,   do_aug=False,
                                batch_size=args.test_batch_size, shuffle=False,
                                num_workers=args.num_workers, cuda=False,
                                out_tensor=True, data_transforms=normalizer)

    return train_loader, val_loader

def create_cifar10_data_loader():
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./cifar10-data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)

    testset = datasets.CIFAR10(root='./cifar10-data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=args.train_batch_size, shuffle=False, num_workers=0)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader, test_loader

def create_open_data_loader():
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    trainset = datasets.ImageFolder(root='./open-images', transform=transform_train)
    train_sampler = DistributedSampler(trainset) \
                        if args.distributed else RandomSampler(trainset)
                            
    train_loader = DataLoader(trainset, batch_size=args.train_batch_size, sampler=train_sampler, num_workers=4)
    print("%d images loaded from open" %len(trainset))
    return train_loader, None
     
# overload MNIST to load the mnist-rot dataset
# when is_rot=True, orig_root specifies the directory of original MNIST images
class MNIST2(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 is_rot=False, out_rot_angle=False,
                 orig_root=None, out_orig_image=False):
        super(datasets.MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set
        self.is_rot = is_rot
        self.out_rot_angle = out_rot_angle
        self.orig_root = orig_root
        self.out_orig_image = out_orig_image
        if self.out_orig_image and self.orig_root is None:
            print("orig_root has to be specified in order to out_orig_image")
            exit(1)

        if not self._check_exists():
            raise RuntimeError("Dataset not found at '%s'." %root)

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        if is_rot:
            self.data, self.targets, self.ground_rot_angles, self.orig_indices = torch.load(os.path.join(self.processed_folder, data_file))
            if self.orig_root:
                self.orig_processed_folder = os.path.join(self.orig_root, 'MNIST', 'processed')
                self.orig_data, self.orig_targets = \
                    torch.load(os.path.join(self.orig_processed_folder, data_file))
        else:
            self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'MNIST', 'processed')

    def __getitem__(self, index):
        if self.is_rot and self.out_rot_angle:
            image, target = super(MNIST2, self).__getitem__(index)
            ground_rot_angle   = self.ground_rot_angles[index]
            orig_index = self.orig_indices[index]
            if self.out_orig_image:
                assert self.orig_targets[orig_index] == target
                orig_image = Image.fromarray(self.orig_data[orig_index].numpy(), mode='L')
                if self.transform is not None:
                    orig_image = self.transform(orig_image)
                return image, target, ground_rot_angle, orig_image
            else:
                return image, target, ground_rot_angle
        else:
            return super(MNIST2, self).__getitem__(index)

def create_mnist_data_loader(normalizing_mean_std, using_mnist_rot=False,
                             out_rot_angle=False, out_orig_image=False):
    transforms_list = [
                      transforms.Resize((56, 56)),
                      # repeat the single channel three times to make it "RGB"
                      transforms.Grayscale(3),
                      transforms.ToTensor(),
                      ]

    if normalizing_mean_std:
        transforms_list.append( transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)) )

    normalizer = transforms.Compose(transforms_list)

    rot_root = "../data-rot"
    orig_root  = "../data"
    if using_mnist_rot:
        mnist_data_root = rot_root
    else:
        mnist_data_root = orig_root

    # MNIST is small data. no need to use multiple workers
    train_loader = DataLoader(
        MNIST2(mnist_data_root, train=True,  transform=normalizer,
               is_rot=using_mnist_rot, out_rot_angle=out_rot_angle,
               orig_root=orig_root, out_orig_image=True),
        batch_size=args.train_batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        MNIST2(mnist_data_root, train=False, transform=normalizer,
               is_rot=using_mnist_rot, out_rot_angle=out_rot_angle,
               orig_root=orig_root, out_orig_image=True),
        batch_size=args.test_batch_size,  shuffle=False)

    return train_loader, val_loader

def init_optimizer(plugin_params, param_to_moduleName, num_train_opt_steps, rot_cls_lr_scale):
    if args.enable_lenses or args.using_xlayer_resnet:
        optimized_params = list( param for param in plugin_params if param[1].requires_grad )
        no_grad_sig     = ['orig_conv']
        no_decay_sig    = ['bias']
        slow_decay_sig  = ['interp_weight']
        rot_cls_sig     = ['rot_cls']
        bn_no_decay_sig = ['bias', 'weight']
        no_grad_names = []
        slow_decay_params = []
        slow_decay_names = []
        no_decay_params = []
        no_decay_names = []
        normal_decay_params = []
        normal_decay_names = []
        rot_cls_params = []
        rot_cls_names = []
        
        for n, p in optimized_params:
            if any(nd in n for nd in rot_cls_sig):
                rot_cls_names.append(n)
                rot_cls_params.append(p)
                continue
            if any(nd in n for nd in no_grad_sig):
                no_grad_names.append(n)
                continue
            if any(nd in n for nd in slow_decay_sig):
                slow_decay_names.append(n)
                slow_decay_params.append(p)
                continue
            if any(nd in n for nd in no_decay_sig) or \
              (param_to_moduleName[p] == 'BatchNorm2d' and any(nd in n for nd in bn_no_decay_sig)):
                no_decay_params.append(p)
                no_decay_names.append(n)
            else:
                normal_decay_params.append(p)
                normal_decay_names.append(n)
                
        print0("Skipped params:")
        print0(no_grad_names)
        print0("Params without weight decay:")
        print0(no_decay_names)
        print0("Params with slow weight decay:")
        print0(slow_decay_names)
        print0("Params with normal weight decay:")
        print0(normal_decay_names)
        print0("Params of rot_cls:")
        print0(rot_cls_names)

        grouped_params = \
            [
                { 'params': normal_decay_params, 'weight_decay': args.weight_decay, 'lr': args.lr },
                { 'params': slow_decay_params, 'weight_decay': args.weight_decay * 0.1, 'lr': args.lr },
                { 'params': no_decay_params, 'weight_decay': 0.0, 'lr': args.lr },
                { 'params': rot_cls_params, 'weight_decay': args.weight_decay, 'lr': args.lr * rot_cls_lr_scale }
            ]
        
        # Backup initial lr. For adjuste_learning_rate()
        for group in grouped_params:
            group['lr0'] = group['lr']
            
        sgd_optimizer    = torch.optim.SGD(grouped_params, momentum=0.9)
        adam_optimizer   = torch.optim.Adam(grouped_params)
        ranger_optimizer = Ranger(grouped_params)
        
    else:
        # placeholder
        dummy_param = [ torch.FloatTensor(1).cuda() ]
        sgd_optimizer    = torch.optim.SGD(dummy_param, lr=args.lr, momentum=0.9)
        adam_optimizer   = torch.optim.Adam(dummy_param, lr=args.lr)
        ranger_optimizer = Ranger(dummy_param, lr=args.lr)

    if args.optimizer == 'sgd':
        optimizer = sgd_optimizer
    elif args.optimizer == 'adam':
        optimizer = adam_optimizer
    elif args.optimizer == 'ranger':
        optimizer = ranger_optimizer

    return optimizer

def adjust_learning_rate(optimizer, frac_epochs):
    global old_decay_count
    decay_count = int(frac_epochs / args.lr_decay_epoch_step)
    if decay_count > old_decay_count:
        lr_decay = args.gamma ** decay_count
        print0("Decay LR by %.2E" %lr_decay)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr0'] * lr_decay
        
    old_decay_count = decay_count
    
def init_amp(net, optimizer):
    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    net, optimizer = amp.initialize(net, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If net = DDP(net) is called
    # before net, ... = amp.initialize(net, ...), the call to amp.initialize may alter
    # the types of net's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # net = DDP(net)
        # delay_allreduce delays all communication to the end of the backward pass.
        net = DDP(net, delay_allreduce=True)
        real_model = net.module
    else:
        real_model = net

    return real_model, net, optimizer

def load_host_checkpoint(host_model, host_cp_filepath):
    if os.path.isfile(host_cp_filepath):
        print0("=> loading checkpoint '{}'... ".format(host_cp_filepath))
        checkpoint = torch.load(host_cp_filepath, map_location = lambda storage, loc: storage.cuda(args.gpu))
        best_prec1 = checkpoint['best_prec1']
        host_model.load_state_dict(checkpoint['state_dict'], strict=False)
        print0("Loaded. Best accuracy: %.1f%%" %(best_prec1))
    else:
        print0("!!! no checkpoint found at '{}'".format(host_cp_filepath))
        exit(0)

def load_plugin_checkpoint(plugin, optimizer):
    match = re.search('([0-9a-z]+)-([0-9a-z,]+)-([0-9]+)-([0-9]+).pth', args.lenses_cp_filepath)
    if match is not None:
        architecture = match.group(1)
        host_stage_name = match.group(2)
        if args.host_stage_name != host_stage_name:
            print0("argument --stage '%s' != cp '%s'" %(args.host_stage_name, host_stage_name))
            exit(0)
        finished_epochs = int( match.group(3) )
        finished_iters = int( match.group(4) )
    else:
        pdb.set_trace()

    checkpoint = torch.load(args.lenses_cp_filepath, map_location = lambda storage, loc: storage.cuda(args.gpu))
    resumed_epoch = checkpoint['epoch']
    if 'world_size' in checkpoint:
        cp_world_size = checkpoint['world_size']
    else:
        cp_world_size = 2
    plugin.load_state_dict(checkpoint['state_dict'])
    if not args.eval_only:
        if checkpoint['opt_name'] == args.optimizer:
            optimizer.load_state_dict(checkpoint['opt_state'])
            print("Optimizer states loaded")
        else:
            print0("cp opt '%s' != current opt '%s'. Skip loading optimizer states" % \
                    (checkpoint['opt_name'], args.optimizer))

    print0("Checkpoint '%s' loaded into '%s'" %(args.lenses_cp_filepath, args.host_stage_name))
    return resumed_epoch, finished_iters, cp_world_size

def save_checkpoint(state, filename, model):
    torch.save(state, filename)
    print("Saved model to '%s'" %filename)
    print_maxweight(model)

bn_counts = [0, 0]
def count_bn(m):
    global bn_counts
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        bn_counts[m.training] += 1

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

# rot_angles: a torch tensor
# Use global variables to do grouping: rot_group_range2index, rot_group_ranges
# rot_group_ranges: 1000*5*2 for mnist
def group_images_by_rot_angle(rot_angles):
    global rot_group_range2index, rot_group_ranges, num_rot_groups
    rot_group_ranges2 = rot_group_ranges[:len(rot_angles)]
    lb_satisfied = rot_angles.view(-1, 1) >= rot_group_ranges2[:, :, 0]
    ub_satisfied = rot_angles.view(-1, 1) <  rot_group_ranges2[:, :, 1]
    range_indices = (lb_satisfied * ub_satisfied).float().argmax(dim=1)
    range_indices2 = rot_group_range2index[range_indices]

    allrot_data_indices = []
    for i in range(num_rot_groups):
        rot_data_indices = torch.masked_select( torch.arange(len(rot_angles)).cuda(), range_indices2==i )
        allrot_data_indices.append(rot_data_indices)

    return allrot_data_indices, range_indices2

# we can set enable_lenses=False, to evaluate the original resnet performance
def train_eval_lenses(epoch, resumed_epoch, finished_iters, cp_world_size,
                      net, real_model, data_loader, optimizer,
                      lenskit, lenses_spec, eval_only, enable_lenses):
    phase_names = ['train', 'test']
    num_lenses = len(lenses_spec)
    crossEnt_ind = nn.CrossEntropyLoss(reduction='none')
    crossEnt = nn.CrossEntropyLoss()
    
    # is_lenskit_frozen = eval_only, but easier to understand from the optimization perspective
    if eval_only:
        run_name = "eval-only"
        is_lenskit_frozen = True
    else:
        run_name = "train-test"
        is_lenskit_frozen = False

    # in resnet.py, enable/disable going through lenses
    real_model.set_pass_featlens(True)

    # If doing_instance_norm, call lenskit.train() to change the BN behavior to instance norm
    if eval_only and not args.doing_instance_norm:
        lenskit.eval()
    else:
        lenskit.train()
        # fix orig_bn (part of orig_conv now) for fair evaluation
        if lenskit.orig_conv.bn:
            lenskit.orig_conv.bn.eval()
    lenskit.frozen = is_lenskit_frozen

    start = time.time()
    data_iter = iter(data_loader)
    iters_per_epoch = len(data_loader)

    if not eval_only and epoch == resumed_epoch and finished_iters > 0:
        # only one batch is left. skip them and start from the next epoch
        if finished_iters * cp_world_size >= iters_per_epoch * args.world_size - args.train_batch_size:
            epoch += 1
            do_skipping = False
            print0("Resume from epoch %d" %epoch)
        elif args.skip_cp_iters:
            do_skipping = True
            print0("Skip %d finished iterations..." %(finished_iters))
    else:
        do_skipping = False

    if args.local_rank == 0:
        # start a new board
        board = SummaryWriter()
        print("Epoch %d %s" %(epoch, run_name))

    lenses_batch_time_stat = AverageMeters()
    batch_time_stat = AverageMeters()
    disp_lenses_correct_insts_t1 = np.zeros((2, num_lenses))
    disp_lenses_correct_insts_t5 = np.zeros((2, num_lenses))
    lenses_all_correct_insts_t1  = np.zeros((2, num_lenses))
    lenses_all_correct_insts_t5  = np.zeros((2, num_lenses))
    lenses_all_meanfeat_corr = np.zeros((2, num_lenses))
    lenses_all_allfeat_corr  = np.zeros((2, num_lenses))
    lenses_all_high_act_corr = np.zeros((2, num_lenses))
    lenses_all_guess_rot_losses    = np.zeros((2, num_lenses))
    lenses_all_correct_rot_guesses = np.zeros((2, num_lenses))
    
    DELTA = 0.001
    # The total numbers of instances of some lenses might be 0. 
    # Add DELTA=0.01 to avoid division by 0
    disp_lenses_total_insts = np.ones((2, num_lenses)) * DELTA
    lenses_total_insts = np.zeros((2, num_lenses))

    disp_lenses_cls_loss = np.zeros((2, num_lenses))
    disp_lenses_meanfeat_corr = np.zeros((2, num_lenses))
    disp_lenses_high_act_corr = np.zeros((2, num_lenses))
    disp_lenses_allfeat_corr  = np.zeros((2, num_lenses))
    disp_lenses_top_activ_loss = np.zeros((2, num_lenses))
    disp_lenses_whole_activ_loss = np.zeros((2, num_lenses))
    disp_lenses_guess_rot_losses = np.zeros((2, num_lenses))
    disp_lenses_correct_rot_guesses = np.zeros((2, num_lenses))
    disp_interval = 0

    # for mnist or stratified Imagenet, num_input_lens_iters = { 0: 'orig', 1: 'strata' }
    # for one by one of ImageNet, num_input_lens_iters = { 0, .., N-1 }, i.e. actual lens indices
    # input_lens_iter_idx is not a lens index when 'mnist' and input_lens_iter_idx == 1.
    # That's why input_lens_iter_idx is not named as input_iter_lens_idx
    # If imagenet and has scaling lenses, can only train/test lenses one by one. 
    # In all other cases, input images of iteration 1 are strata of different rotations.
    has_input_strata = (args.dataset != 'imagenet' and args.dataset != 'cifar10') or args.rot_lenses_only
    if has_input_strata:
        num_input_lens_iters = 2
    else:
        num_input_lens_iters = num_lenses

    if args.guessing_rot_lenses:
        old_rot_w = lenskit.rot_cls.cls.weight.detach().clone()
    # Save model at every LR decay.
    args.save_model_interval = int(iters_per_epoch * args.lr_decay_epoch_step)
    print0("Models will be saved to '%s' every %d iterations" %(args.cp_dir, args.save_model_interval))
    
    for step in range(iters_per_epoch):
        frac_epoch = (step+1.0) / iters_per_epoch
        batch_data = next(data_iter)
        batch_data = [ t.cuda() for t in batch_data ]

        # skip some iterations on which the checkpoint was trained previously
        if do_skipping:
            if step < finished_iters:
                if (step+1) % 50 == 0:
                    print0(step+1, end=" ", flush=True)
                continue
            else:
                print0("Done. Resume training")
                do_skipping = False

        im_data0 = batch_data[0]
        batch_labels = batch_data[1]

        for input_lens_iter_idx in range(num_input_lens_iters):
            if args.dataset == 'imagenet' or args.dataset == 'cifar10':
                if (not has_input_strata) or input_lens_iter_idx == 0:
                    curr_lens_name, img_trans = lenses_spec[input_lens_iter_idx][:2]
                    # For lens 0 (orig lens), img_trans is a do-nothing Sequential, i.e. no transformation
                    # For other lenses, img_trans does the corresponding geometric transformation
                    im_data1 = img_trans(im_data0)
                    all_ground_lens_strata_dindices = [ [] for i in range(num_lenses) ]
                    all_ground_lens_strata_dindices[input_lens_iter_idx] = torch.arange(len(im_data0)).cuda()
                    all_ground_lens_indices = [input_lens_iter_idx]
                    dindices_are_stratified = False
                    ground_rot_labels = torch.ones(len(im_data0), dtype=torch.long).cuda() * input_lens_iter_idx
                else:
                    # mixture of image transformations to be handled by different lenses
                    curr_lens_name = 'strata' 
                    random_rot_labels = torch.randint(4, (len(im_data0),)).cuda()
                    random_rot_angles = random_rot_labels * 90
                    all_ground_lens_strata_dindices, random_rot_labels2 = group_images_by_rot_angle(random_rot_angles)
                    assert (random_rot_labels != random_rot_labels2).sum() == 0
                    ground_rot_labels = random_rot_labels
                    all_ground_lens_indices = list(range(num_lenses))
                    dindices_are_stratified = True
                    
                    im_data1 = torch.zeros_like(im_data0)
                    for lens_idx in range(num_lenses):
                        stratum_dindices = all_ground_lens_strata_dindices[lens_idx]
                        lens_name, img_trans = lenses_spec[lens_idx][:2]
                        im_stratum = img_trans(im_data0[stratum_dindices])
                        im_data1.index_copy_(0, stratum_dindices, im_stratum)
                    
            elif args.dataset == 'mnist':
                ground_rot_angles  = batch_data[2]
                orig_images = batch_data[3]
                if input_lens_iter_idx == 0:
                    curr_lens_name = 'orig'
                    im_data1 = orig_images
                    # all images belong to lens 0
                    all_ground_lens_strata_dindices = [ torch.arange(len(im_data0)).cuda(), [] ]
                    all_ground_lens_indices = [0]
                    dindices_are_stratified = False
                    ground_rot_labels = torch.zeros(len(im_data0), dtype=torch.long).cuda()
                else:
                    # if input_lens_iter_idx == 1, images are already stratified, i.e., rotated differently
                    # lenskit.curr_lens_name will be updated as 's%d' in the lenskit strata loop
                    # mixture of image transformations for different lenses
                    curr_lens_name = 'strata' 
                    im_data1 = im_data0
                    # all_ground_lens_strata_dindices: a list containing four sub-lists of differently rotated images
                    # ground_rot_labels: the ground lens labels of each image
                    all_ground_lens_strata_dindices, ground_rot_labels = group_images_by_rot_angle(ground_rot_angles)
                    # Probably more than one ground lens is involved. 
                    # Data strata indices are specified in all_ground_lens_strata_dindices
                    all_ground_lens_indices = list(range(num_lenses))
                    dindices_are_stratified = True
            else:
                pdb.set_trace()
                
            lenskit.input_lens_iter_idx     = input_lens_iter_idx
            lenskit.all_ground_lens_strata_dindices  = all_ground_lens_strata_dindices
            lenskit.all_ground_lens_indices = all_ground_lens_indices
            lenskit.curr_lens_name          = curr_lens_name
            lenskit.dindices_are_stratified = dindices_are_stratified
            
            # phase=0: training, phase=1: test,
            # if eval_only = True, training phase means to use original feature maps for prediction
            # (lenses are disabled)
            # Test first, then train. Otherwise the model first sees the data \
            # during training, and evaluate on the same data during test, which is cheating.
            # In training phase, train lenses and evaluate the original performance.
            # Classify original features of transformed images
            # In test phase,     evaluate lenses.
            # Classify lens-transformed features of transformed images
            # But for lens 0, training and test is the same: original images through original pipeline
            for phase in (1, 0):    # 1: test, 0: train
                phase_name = phase_names[phase]
                lenskit.phase_name = phase_name

                if phase_name == 'train':
                    # using_lenses = False: do not use lenses to transform features for downstream.
                    # But still use lenses to transform features for alignment training
                    using_lenses = False
                    # During training phase, never use guessed lens to transform features
                    using_guessed_lenses = False
                    if eval_only:
                        updating_lens = False
                    else:
                        updating_lens = True
                else:
                    # we can set args.enable_lenses = False, to evaluate original resnet
                    using_lenses = args.enable_lenses
                    updating_lens = False
                    # If guessing lens, then in test phase, use guessed lenses to transform for downstream.
                    # If input_lens_iter_idx == 0 (all images are original), 
                    # lenskit extracts and caches original features.
                    # If has_input_strata at the same time, RotClassifier doesn't make guesses.
                    if has_input_strata:
                        using_guessed_lenses = args.guessing_rot_lenses and input_lens_iter_idx != 0
                    else:
                        using_guessed_lenses = args.guessing_rot_lenses
                        
                lenskit.using_lenses  = using_lenses
                lenskit.updating_lens = updating_lens
                lenskit.using_guessed_lenses = using_guessed_lenses
                
                # Gradients are not needed for the host feature extractor. This is to save RAM
                with torch.no_grad():
                    cls_scores = net(im_data1)

                # losses for each image, instead of the averaged loss
                # cls_losses is not used for optimization. Just for reference
                cls_losses = crossEnt_ind(cls_scores, batch_labels)
                cls_loss = crossEnt(cls_scores, batch_labels)

                pred_labels_t1 = cls_scores.argmax(dim=1)
                pred_labels_t5 = cls_scores.topk(5, dim=1)[1]
                is_t1_correct = ( pred_labels_t1 == batch_labels )
                is_t5_correct = ( pred_labels_t5 == batch_labels.view(-1, 1) ).sum(dim=1)

                # args.guessing_rot_lenses implies args.using_rot_lenses
                if args.guessing_rot_lenses:
                    guess_rot_losses = crossEnt_ind(lenskit.rot_scores, ground_rot_labels)
                    guess_rot_loss   = crossEnt(lenskit.rot_scores, ground_rot_labels)
                    is_guessed_rot_correct = ( lenskit.guessed_rot_labels == ground_rot_labels )
                    
                for stratum_ground_lens_idx in all_ground_lens_indices:
                    # Calculate lens-wise accuracy by grouping by groundtruth lens assignments.
                    # So we use all_ground_lens_strata_dindices[stratum_ground_lens_idx].
                    glens_stratum_dindices = all_ground_lens_strata_dindices[stratum_ground_lens_idx]
                    stratum_glens_name = lenses_spec[stratum_ground_lens_idx][0]
                    if len(glens_stratum_dindices) == 0:
                        continue

                    if epoch == resumed_epoch and step == 0 and args.enable_lenses:
                        print0("%s Data: %s, '%s' in: %s, out: %s" % \
                               (phase_name, list(im_data1.shape), stratum_glens_name,
                                lenskit.input_feat_shape,
                                lenskit.feat_shapes[stratum_ground_lens_idx]))

                    lens_is_t1_correct = is_t1_correct[glens_stratum_dindices]
                    lens_is_t5_correct = is_t5_correct[glens_stratum_dindices]
                    lens_cls_loss = cls_losses[glens_stratum_dindices].mean()
                    
                    if args.distributed:
                        reduced_correct_insts_t1 = reduce_tensor( lens_is_t1_correct.sum() )
                        reduced_correct_insts_t5 = reduce_tensor( lens_is_t5_correct.sum() )
                        # .data has no grad, so reduced_cls_loss is only for showing stats, not for training
                        reduced_cls_loss = reduce_tensor(lens_cls_loss.data)
                    else:
                        reduced_correct_insts_t1 = lens_is_t1_correct.sum()
                        reduced_correct_insts_t5 = lens_is_t5_correct.sum()
                        reduced_cls_loss = lens_cls_loss.data

                    # sum of cls_losses over a disp_interval.
                    # for computing the average loss during this interval
                    disp_lenses_cls_loss[phase, stratum_ground_lens_idx] += reduced_cls_loss.item()

                    disp_lenses_correct_insts_t1[phase, stratum_ground_lens_idx] += reduced_correct_insts_t1.item()
                    disp_lenses_correct_insts_t5[phase, stratum_ground_lens_idx] += reduced_correct_insts_t5.item()
                    disp_lenses_total_insts[phase, stratum_ground_lens_idx] += len(glens_stratum_dindices)

                    if args.guessing_rot_lenses:
                        lens_is_guessed_rot_correct = is_guessed_rot_correct[glens_stratum_dindices]
                        lens_guess_rot_loss = guess_rot_losses[glens_stratum_dindices].sum()
                        if args.distributed:
                            reduced_correct_rot_guesses = reduce_tensor( lens_is_guessed_rot_correct.sum() )
                            reduced_lens_guess_rot_loss = reduce_tensor(lens_guess_rot_loss.data)
                        else:
                            reduced_correct_rot_guesses = lens_is_guessed_rot_correct.sum()
                            reduced_lens_guess_rot_loss = lens_guess_rot_loss.data
                        
                        disp_lenses_correct_rot_guesses[phase, stratum_ground_lens_idx] += reduced_correct_rot_guesses.item()
                        disp_lenses_guess_rot_losses[phase, stratum_ground_lens_idx]    += reduced_lens_guess_rot_loss.item()

                # When all_strata_lens_indices = [0], top_activ_loss = whole_activ_loss = tensor(0.).
                # So in this case, if guessing_rot_lenses, then total_loss = guess_rot_loss
                total_activ_loss = (1 - args.whole_act_loss_weight) * lenskit.top_activ_loss \
                                      + args.whole_act_loss_weight  * lenskit.whole_activ_loss

                # total_loss will only be optimized during training phase. 
                # So no need to decide whether it's training or test here.
                # If has_input_strata and input_lens_iter_idx == 0, the input data all belong to lens 0. 
                # Input too biased for rot_cls training. So skip this lens iteration
                training_rot_cls = args.guessing_rot_lenses and not (has_input_strata and input_lens_iter_idx == 0)
                if training_rot_cls:
                    total_loss = total_activ_loss + guess_rot_loss
                else:
                    total_loss = total_activ_loss
                    
                for stratum_lens_idx in lenskit.all_strata_lens_indices:
                    # Calculate lens-wise losses of each lens.
                    # In the lenskit, the losses are calculated by iterating all_strata_lens_indices.
                    # It's different from all_ground_lens_indices if using_rot_lenses & using_guessed_lenses, i.e., 
                    # when testing on ImageNet rotations, and using_guessed_lenses.
                    # In that case, all_ground_lens_indices = [i], all_strata_lens_indices = [0, ..., N-1].
                    # This inaccuracy is unavoidable, so let it be.
                    # disp_lenses_top_activ_loss and disp_lenses_whole_activ_loss should be the accumulated losses.
                    # But it's difficult to track the total losses per ground lens. 
                    # So just record the average predicted lens loss here. Later disp_lenses_top_activ_loss 
                    # and disp_lenses_whole_activ_loss are averaged by iterations instead of number of instances
                    # A bit inaccurate, but much easier to implement.
                    disp_lenses_top_activ_loss[phase, stratum_lens_idx] += lenskit.all_lenses_top_activ_loss[stratum_lens_idx]
                    disp_lenses_whole_activ_loss[phase, stratum_lens_idx] += lenskit.all_lenses_whole_activ_loss[stratum_lens_idx]

                # phase_name == 'train' and not eval_only => net.updating_lens = True
                # if input_lens_iter_idx == 0 and not using_guessed_lenses, it's the identity/orig lens, 
                # so no training on activation loss is performed. 
                # In this case, total_activ_loss = tensor(0.), so no need to 
                # test whether input_lens_iter_idx == 0.
                # But would still optimize guess_rot_loss if args.guessing_rot_lenses.
                if not eval_only and phase_name == 'train':
                    optimizer.zero_grad()
                    with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    
                    def check_grad(training_rot_cls, rot_w): 
                        if training_rot_cls:
                            rot_grad = rot_w.grad
                            rot_w_delta = (rot_w - old_rot_w).abs().sum().item()
                            print0("rot w delta %.1f, rot grad max %.1f, mean %.1f" %(rot_w_delta, rot_grad.max().item(), rot_grad.abs().mean().item()))
                   
                    # gradient clipping doesn't work, for unknown reasons
                    if False and args.max_grad_value > 0:
                        check_grad(training_rot_cls, lenskit.rot_cls.cls.weight)         
                        if args.amp:
                            params = amp.master_params(optimizer)
                        else:
                            params = optimizer.params()
                        torch.nn.utils.clip_grad_value_(params, args.max_grad_value)
                    if training_rot_cls:
                        check_grad(training_rot_cls, lenskit.rot_cls.cls.weight)
                    if training_rot_cls:
                        old_rot_w = lenskit.rot_cls.cls.weight.detach().clone()
                        
                    optimizer.step()

                    adjust_learning_rate(optimizer, epoch + frac_epoch)

                # When "not frozen" (not eval_only) and training,
                # get_top_activations() is mandatory for computing losses.
                # During test phase, get_top_activations() is optional. But if "not frozen",
                # we still compute feature correlations to see how it improves due to training
                # (test happens before training, so correlations in training phase should be larger)
                # When eval_only, we still want to get feature correlations for bookkeeping.
                # But as lenses are frozen, training/test correlations are the same. 
                # So only need to compute once (i.e., during test)
                getting_top_act = (not is_lenskit_frozen) or (phase_name == 'test')
                if getting_top_act and input_lens_iter_idx != 0:
                    # Compute the activation correlations, to know the improvement.
                    # In order to compute lens-specific loss, we need to update
                    # lenskit.stratum_lens_idx and lenskit.lens_stratum_dindices in each iteration.
                    # They are updated by strata_iterator / lenskit.strata_iter_obj
                    strata_iterator = iter(lenskit.strata_iter_obj)
                    for stratum_iter_idx in strata_iterator:
                        lens_stratum_batch_size = len(lenskit.lens_stratum_dindices)
                        # stratum_lens_idx == 0 is possible even if input_lens_iter_idx !=0.
                        if lenskit.stratum_lens_idx == 0 or lens_stratum_batch_size == 0:
                            continue

                        lenskit.contrast_feat_low_high('new', do_compute_loss=False, verbose=True)
                        # disp_lenses_meanfeat_corr, disp_lenses_high_act_corr, and disp_lenses_allfeat_corr always store the 'new' correlations.
                        # 'old' correlations are printed inside lenskit.forward() and forgotten
                        disp_lenses_meanfeat_corr[phase, lenskit.stratum_lens_idx] += lenskit.meanfeat_corr
                        disp_lenses_high_act_corr[phase, lenskit.stratum_lens_idx] += lenskit.high_act_corr
                        disp_lenses_allfeat_corr[phase, lenskit.stratum_lens_idx]  += lenskit.allfeat_corr

                end = time.time()
                batch_time = end - start
                # lenses_batch_time_stat is an AverageMeters, 
                # and lenses_batch_time_stat.val is a dictionary. 
                # Note when on 'mnist' and input_lens_iter_idx == 1, it stores the processing time
                # of the stratified images, instead of lens 1.
                # As we couldn't track the time spent on each stratum, we allocate the time 
                # proportional to the fraction of the stratum in the whole batch.
                all_ground_lens_strata_batch_size = [ len(stratum) for stratum in all_ground_lens_strata_dindices ]
                print0("%d-%d: grd %s, use %s" %(phase, input_lens_iter_idx, 
                                                 all_ground_lens_strata_batch_size, 
                                                 list(lenskit.all_lens_strata_batch_sizes)))
                strata_size_frac = lenskit.all_lens_strata_batch_sizes.astype(float) / lenskit.all_lens_strata_batch_sizes.sum()
                batch_time_stat.update(phase, batch_time)
                for stratum_lens_idx in range(num_lenses):
                    stratum_batch_time_frac = batch_time * strata_size_frac[stratum_lens_idx]
                    lenses_batch_time_stat.update((phase, stratum_lens_idx), stratum_batch_time_frac)
                    
                start = time.time()
            # end of phase loop
        # end of input lens loop
        if args.enable_lenses:
            lenskit.clear_cache()
        disp_interval += 1
        
        if (step+1) % args.disp_interval == 0 or (step+1) == iters_per_epoch:
            lenses_all_meanfeat_corr += disp_lenses_meanfeat_corr
            lenses_all_high_act_corr += disp_lenses_high_act_corr
            lenses_all_allfeat_corr     += disp_lenses_allfeat_corr
            lenses_all_guess_rot_losses += disp_lenses_guess_rot_losses
            lenses_all_correct_rot_guesses += disp_lenses_correct_rot_guesses
            
            disp_lenses_cls_loss    /= disp_interval
            disp_lenses_top_activ_loss /= disp_interval
            disp_lenses_whole_activ_loss /= disp_interval
            disp_lenses_meanfeat_corr /= disp_interval
            disp_lenses_high_act_corr /= disp_interval
            disp_lenses_allfeat_corr /= disp_interval
            disp_lenses_correct_acc_t1 = disp_lenses_correct_insts_t1 / disp_lenses_total_insts
            disp_lenses_correct_acc_t5 = disp_lenses_correct_insts_t5 / disp_lenses_total_insts
            disp_lenses_guess_rot_losses = disp_lenses_guess_rot_losses / disp_lenses_total_insts
            disp_lenses_rot_guess_acc = disp_lenses_correct_rot_guesses / disp_lenses_total_insts
            
            for phase in range(2):
                phase_name = phase_names[phase]
                imgs_per_sec = args.world_size * len(im_data0) / batch_time_stat.avg[phase]
                for input_lens_idx in range(num_lenses):
                    lens_name = lenses_spec[input_lens_idx][0]
                    # When using guessed lenses, the loss and corr values are not indexed by input_lens_idx,
                    # but by stratum_lens_idx, each of which corresponds to guessed lens assignments.
                    # This semantic discrepancy should be kept in mind when examining these values.
                    # Even if the features are well recovered, if guessed lens assignments have many errors,
                    # the loss values will still be high and corr values be low.
                    # Accuracy scores are correctly indexed by input_lens_idx.
                    top_activ_loss_str   = "%.2f" %(disp_lenses_top_activ_loss[phase, input_lens_idx])
                    whole_activ_loss_str   = "%.2f" %(disp_lenses_whole_activ_loss[phase, input_lens_idx])
                    meanfeat_corr_str    = "%.3f" %(disp_lenses_meanfeat_corr[phase, input_lens_idx])
                    high_act_corr_str    = "%.3f" %(disp_lenses_high_act_corr[phase, input_lens_idx])
                    feat_corr_str        = "%.3f" %(disp_lenses_allfeat_corr[phase, input_lens_idx])
                    print0("%3d/%3d t %.3f s %d, %s %s cl: %.2f, acc: %.3f/%.3f" \
                            % (step+1, iters_per_epoch, lenses_batch_time_stat.avg[phase, input_lens_idx],
                               imgs_per_sec, phase_name, lens_name,
                               disp_lenses_cls_loss[phase, input_lens_idx],
                               disp_lenses_correct_acc_t1[phase, input_lens_idx],
                               disp_lenses_correct_acc_t5[phase, input_lens_idx]))
                        
                    output_corr = (not is_lenskit_frozen or phase_name == 'test') and (input_lens_idx != 0)
                    output_activ_loss = (not is_lenskit_frozen) and (input_lens_idx != 0)
                    
                    # if eval_only or (not is_lenskit_frozen) and test, activation losses
                    # are not computed. so losses are always 0
                    # but when eval_only and test, feature correlations are still computed
                    # so output_activ_loss implies output_corr, but not vice versa
                    if output_corr:
                        print0("top act: %s, all act: %s, high cor: %s, mean cor: %s, cor: %s" % \
                                (top_activ_loss_str, whole_activ_loss_str,
                                 high_act_corr_str, meanfeat_corr_str, feat_corr_str))

                    # Even if the strata are original images, still output guess_rot_losses
                    # 'train' and 'test' rot losses are the same (rot_cls updated after 'train', 
                    # and then not used until next iteration). No need to output twice.
                    if phase_name == 'test' and args.guessing_rot_lenses:
                        print0("%d: rot loss: %.3f, acc: %.3f" \
                                        %(input_lens_idx, 
                                          disp_lenses_guess_rot_losses[phase, input_lens_idx], 
                                          disp_lenses_rot_guess_acc[phase, input_lens_idx]))
                        
                    if args.local_rank == 0:
                        if output_corr:
                            board.add_scalar('%s-%d-corr-mean' %(phase_name, input_lens_idx), \
                                                disp_lenses_meanfeat_corr[phase, input_lens_idx], step+1)
                            board.add_scalar('%s-%d-corr-high' %(phase_name, input_lens_idx), \
                                                disp_lenses_high_act_corr[phase, input_lens_idx], step+1)
                            board.add_scalar('%s-%d-corr-all' %(phase_name, input_lens_idx),  \
                                                disp_lenses_allfeat_corr[phase, input_lens_idx], step+1)
                        if output_activ_loss:
                            board.add_scalar('%s-%d-loss-top' %(phase_name, input_lens_idx),  \
                                                disp_lenses_top_activ_loss[phase, input_lens_idx], step+1)
                            board.add_scalar('%s-%d-loss-all' %(phase_name, input_lens_idx),  \
                                                disp_lenses_whole_activ_loss[phase, input_lens_idx], step+1)
                        if args.guessing_rot_lenses:
                            board.add_scalar('%s-%d-rot-loss' %(phase_name, input_lens_idx),  \
                                                disp_lenses_guess_rot_losses[phase, input_lens_idx], step+1)
                            board.add_scalar('%s-%d-rot-guess-acc' %(phase_name, input_lens_idx),  \
                                                disp_lenses_rot_guess_acc[phase, input_lens_idx], step+1)
                                                
            disp_lenses_avg_acc_diff_t1 = disp_lenses_correct_acc_t1[1] - disp_lenses_correct_acc_t1[0]
            disp_lenses_avg_acc_diff_t5 = disp_lenses_correct_acc_t5[1] - disp_lenses_correct_acc_t5[0]

            disp_lenses_cls_loss[...] = 0
            disp_lenses_top_activ_loss[...] = 0
            disp_lenses_whole_activ_loss[...] = 0
            
            disp_interval = 0
            lenses_batch_time_stat.reset()
            batch_time_stat.reset()

            lenses_all_correct_insts_t1 += disp_lenses_correct_insts_t1
            lenses_all_correct_insts_t5 += disp_lenses_correct_insts_t5
            lenses_total_insts += disp_lenses_total_insts - DELTA
            lenses_all_avg_acc_diff_t1 = ( lenses_all_correct_insts_t1[1] - lenses_all_correct_insts_t1[0] ) / lenses_total_insts[0]
            lenses_all_avg_acc_diff_t5 = ( lenses_all_correct_insts_t5[1] - lenses_all_correct_insts_t5[0] ) / lenses_total_insts[0]
            
            for input_lens_idx in range(1, num_lenses):
                print0("%-14s %d t1/t5 diff: %.1f/%.1f, overall t1/t5 diff: %.1f/%.1f" %\
                        (lenses_spec[input_lens_idx][0], step+1,
                         disp_lenses_avg_acc_diff_t1[input_lens_idx]*100, 
                         disp_lenses_avg_acc_diff_t5[input_lens_idx]*100,
                         lenses_all_avg_acc_diff_t1[input_lens_idx]*100,  
                         lenses_all_avg_acc_diff_t5[input_lens_idx]*100))

                if args.local_rank == 0:
                    board.add_scalar('%d-t1-diff' %(input_lens_idx),  \
                                        disp_lenses_avg_acc_diff_t1[input_lens_idx]*100, step+1)
                    board.add_scalar('%d-t5-diff' %(input_lens_idx),  \
                                        disp_lenses_avg_acc_diff_t5[input_lens_idx]*100, step+1)

            disp_lenses_correct_insts_t1[...] = 0
            disp_lenses_correct_insts_t5[...] = 0
            disp_lenses_meanfeat_corr[...] = 0
            disp_lenses_high_act_corr[...] = 0
            disp_lenses_allfeat_corr[...] = 0
            disp_lenses_guess_rot_losses[...] = 0
            disp_lenses_correct_rot_guesses[...] = 0
            disp_lenses_total_insts[...] = DELTA
            start = time.time()

        # remember best prec@1 and save checkpoint
        if (not eval_only) and args.local_rank == 0 \
               and ((step+1) % args.save_model_interval == 0 or step+1 == iters_per_epoch):
            cp_filepath = "%s/%s-%s-%d-%04d.pth" %(args.cp_dir, args.architecture, args.host_stage_name, epoch, step+1)
            save_checkpoint({
                    'epoch':      epoch,
                    'arch':       args.architecture,
                    'state_dict': lenskit.state_dict(),
                    'opt_state' : optimizer.state_dict(),
                    'opt_name':   args.optimizer,
                    'world_size': args.world_size
                }, filename=cp_filepath, model=lenskit)

    for input_lens_idx in range(num_lenses):
        avg_acc_t1 = lenses_all_correct_insts_t1[:, input_lens_idx] / lenses_total_insts[0, input_lens_idx]
        avg_acc_t5 = lenses_all_correct_insts_t5[:, input_lens_idx] / lenses_total_insts[0, input_lens_idx]
        lenses_all_rot_guess_acc = lenses_all_correct_rot_guesses / lenses_total_insts[0, input_lens_idx]
        lenses_all_guess_rot_avg_losses = lenses_all_guess_rot_losses / lenses_total_insts[0, input_lens_idx]
        print0("%-14s overall orig t1/t5: %.1f/%.1f, lens t1/t5: %.1f/%.1f" %\
                (lenses_spec[input_lens_idx][0],
                 avg_acc_t1[0]*100, avg_acc_t5[0]*100,
                 avg_acc_t1[1]*100, avg_acc_t5[1]*100))
        print0("%-14s overall feat corr: %.3f/%.3f/%.3f" %\
                (lenses_spec[input_lens_idx][0], 
                 lenses_all_high_act_corr[1, input_lens_idx] / iters_per_epoch,
                 lenses_all_meanfeat_corr[1, input_lens_idx] / iters_per_epoch,
                 lenses_all_allfeat_corr[1, input_lens_idx]  / iters_per_epoch))
        if args.guessing_rot_lenses:
            print0("%d: rot loss: %.3f, acc: %.3f" %\
                    (input_lens_idx, 
                     lenses_all_guess_rot_avg_losses[1, input_lens_idx], 
                     lenses_all_rot_guess_acc[1, input_lens_idx]))
                        

# baseline: resnet (frozen) + an extra layer (trainable)
# 'enable_lenses' is only a placeholder to make the arguments conform to train_eval_lenses()
def train_eval_resnet(epoch, resumed_epoch, finished_iters, cp_world_size,
                      net, real_model, data_loader, optimizer,
                      xlayer, lenses_spec, eval_only, enable_lenses):
    phase_names = ['train', 'test']
    num_tforms = len(lenses_spec)
    crossEnt_ind = nn.CrossEntropyLoss(reduction='none')
    crossEnt = nn.CrossEntropyLoss()
    
    # is_xlayer_frozen = eval_only, but easier to understand from the optimization perspective
    if eval_only:
        run_name = "eval-only"
        xlayer.eval()
        is_xlayer_frozen = True
    else:
        run_name = "train-test"
        is_xlayer_frozen = False

    if eval_only and not args.doing_instance_norm:
        xlayer.eval()
    else:
        xlayer.train()

    start = time.time()
    data_iter = iter(data_loader)
    iters_per_epoch = len(data_loader)

    if not eval_only and epoch == resumed_epoch and finished_iters > 0:
        # only one batch is left. skip them and start from the next epoch
        if finished_iters * cp_world_size >= iters_per_epoch * args.world_size - args.train_batch_size:
            epoch += 1
            do_skipping = False
            print0("Resume from epoch %d" %epoch)
        elif args.skip_cp_iters:
            do_skipping = True
            print0("Skip %d finished iterations..." %(finished_iters))
    else:
        do_skipping = False

    if args.local_rank == 0:
        # start a new board
        board = SummaryWriter()
        print("Epoch %d %s" %(epoch, run_name))

    batch_time_stat = AverageMeters()
    disp_tforms_correct_insts_t1 = np.zeros((2, num_tforms))
    disp_tforms_correct_insts_t5 = np.zeros((2, num_tforms))
    tforms_all_correct_insts_t1  = np.zeros((2, num_tforms))
    tforms_all_correct_insts_t5  = np.zeros((2, num_tforms))
    
    DELTA = 0.001
    # The total numbers of instances of some tforms might be 0. 
    # Add DELTA=0.01 to avoid division by 0
    disp_tforms_total_insts = np.ones((2, num_tforms)) * DELTA
    tforms_total_insts = np.zeros((2, num_tforms))

    disp_tforms_cls_loss = np.zeros((2, num_tforms))
    disp_interval = 0

    # for mnist or stratified Imagenet, num_input_lens_iters = { 0: 'orig', 1: 'strata' }
    # for one by one of ImageNet, num_input_lens_iters = { 0, .., N-1 }, i.e. actual lens indices
    # input_lens_iter_idx is not a lens index when 'mnist' and input_lens_iter_idx == 1.
    # That's why input_lens_iter_idx is not named as input_iter_lens_idx
    # If imagenet and has scaling lenses, can only train/test lenses one by one. 
    # In all other cases, input images of iteration 1 are strata of different rotations.
    has_input_strata = (args.dataset != 'imagenet' and args.dataset != 'cifar10') or args.rot_lenses_only
    if has_input_strata:
        num_input_tform_iters = 2
    else:
        num_input_tform_iters = num_tforms

    # Save model at every LR decay.
    args.save_model_interval = int(iters_per_epoch * args.lr_decay_epoch_step)
    print("Models will be saved to '%s' every %d iterations" %(args.cp_dir, args.save_model_interval))

    for step in range(iters_per_epoch):
        frac_epoch = (step+1.0) / iters_per_epoch
        batch_data = next(data_iter)
        batch_data = [ t.cuda() for t in batch_data ]

        # skip some iterations on which the checkpoint was trained previously
        if do_skipping:
            if step < finished_iters:
                if (step+1) % 50 == 0:
                    print0(step+1, end=" ", flush=True)
                continue
            else:
                print0("Done. Resume training")
                do_skipping = False

        im_data0 = batch_data[0]
        batch_labels = batch_data[1]

        for input_tform_iter_idx in range(num_input_tform_iters):
            if args.dataset == 'imagenet' or args.dataset == 'cifar10':
                if (not has_input_strata) or input_tform_iter_idx == 0:
                    curr_tform_name, img_trans = lenses_spec[input_tform_iter_idx][:2]
                    # For tform 0 (orig lens), img_trans is a do-nothing Sequential, i.e. no transformation
                    # For other tforms, img_trans does the corresponding geometric transformation
                    im_data1 = img_trans(im_data0)
                    all_ground_tform_strata_dindices = [ [] for i in range(num_tforms) ]
                    all_ground_tform_strata_dindices[input_tform_iter_idx] = torch.arange(len(im_data0)).cuda()
                    all_ground_tform_indices = [input_tform_iter_idx]
                    dindices_are_stratified = False
                    ground_rot_labels = torch.ones(len(im_data0), dtype=torch.long).cuda() * input_tform_iter_idx
                else:
                    # mixture of image transformations
                    curr_tform_name = 'strata' 
                    random_rot_labels = torch.randint(4, (len(im_data0),)).cuda()
                    random_rot_angles = random_rot_labels * 90
                    all_ground_tform_strata_dindices, random_rot_labels2 = group_images_by_rot_angle(random_rot_angles)
                    assert (random_rot_labels != random_rot_labels2).sum() == 0
                    ground_rot_labels = random_rot_labels
                    all_ground_tform_indices = list(range(num_tforms))
                    dindices_are_stratified = True
                    
                    im_data1 = torch.zeros_like(im_data0)
                    for lens_idx in range(num_tforms):
                        stratum_dindices = all_ground_tform_strata_dindices[lens_idx]
                        tform_name, img_trans = lenses_spec[lens_idx][:2]
                        im_stratum = img_trans(im_data0[stratum_dindices])
                        im_data1.index_copy_(0, stratum_dindices, im_stratum)
                
            elif args.dataset == 'mnist':
                ground_rot_angles  = batch_data[2]
                orig_images = batch_data[3]

                if input_tform_iter_idx == 0:
                    curr_tform_name = 'orig'
                    im_data1 = orig_images
                    # all images belong to tform 0
                    all_ground_tform_strata_dindices = [ torch.arange(len(im_data0)).cuda(), [] ]
                    all_ground_tform_indices = [0]
                else:
                    curr_tform_name = 'strata' # mixture of data for different tforms
                    im_data1 = im_data0
                    # all_ground_tform_strata_dindices: a list containing four sub-lists of differently rotated images
                    # ground_rot_labels: the ground tform labels of each image
                    all_ground_tform_strata_dindices, ground_rot_labels = group_images_by_rot_angle(ground_rot_angles)
                    # Probably more than one ground tform is involved. 
                    # Data strata indices are specified in all_ground_tform_strata_dindices
                    all_ground_tform_indices = list(range(num_tforms))

            else:
                pdb.set_trace()
                
            # phase=0: training, phase=1: test,
            # Test first, then train. Otherwise the model first sees the data \
            # during training, and evaluate on the same data during test, which is cheating.
            # Classify xlayer-transformed features of transformed images
            # Even for tform 0, training/test is through xlayer-transformed features,
            # as in this pipeline, we don't bother to predict the type of transformation
            for phase in (1, 0):    # 1: test, 0: train
                phase_name = phase_names[phase]

                # resnet.layer4 is wrapped in torch.enable_grad(),
                # so we can still do training of the xlayer.
                # This trick is to greatly reduce the RAM use
                # but the cls_scores returned from net() has no gradient
                # We have to use the stored net.cls_scores
                with torch.no_grad():
                    net(im_data1)
                cls_scores = real_model.cls_scores
                # cls_losses: losses for each image. cls_loss: the averaged loss
                # cls_losses is not used for optimization. Just for computing stats
                cls_losses = crossEnt_ind(cls_scores, batch_labels)
                cls_loss = crossEnt(cls_scores, batch_labels)
                 
                pred_labels_t1 = cls_scores.argmax(dim=1)
                pred_labels_t5 = cls_scores.topk(5, dim=1)[1]
                is_t1_correct = ( pred_labels_t1 == batch_labels )
                is_t5_correct = ( pred_labels_t5 == batch_labels.view(-1, 1) ).sum(dim=1)
                   
                for stratum_ground_tform_idx in all_ground_tform_indices:
                    # Calculate tform-wise accuracy by grouping by groundtruth tform assignments.
                    # So we use all_ground_tform_strata_dindices[stratum_ground_tform_idx].
                    # gtform: ground_tform
                    gtform_stratum_dindices = all_ground_tform_strata_dindices[stratum_ground_tform_idx]
                    stratum_gtform_name = lenses_spec[stratum_ground_tform_idx][0]
                    if len(gtform_stratum_dindices) == 0:
                        continue

                    tform_is_t1_correct = is_t1_correct[gtform_stratum_dindices]
                    tform_is_t5_correct = is_t5_correct[gtform_stratum_dindices]
                    tform_cls_loss = cls_losses[gtform_stratum_dindices].mean()
                    
                    if args.distributed:
                        reduced_correct_insts_t1 = reduce_tensor( tform_is_t1_correct.sum() )
                        reduced_correct_insts_t5 = reduce_tensor( tform_is_t5_correct.sum() )
                        reduced_tform_cls_loss = reduce_tensor(tform_cls_loss.data)
                    else:
                        reduced_correct_insts_t1 = tform_is_t1_correct.sum()
                        reduced_correct_insts_t5 = tform_is_t5_correct.sum()
                        reduced_tform_cls_loss = tform_cls_loss.data

                    # sum of cls_losses over a disp_interval.
                    # for computing the average loss during this interval
                    disp_tforms_cls_loss[phase, stratum_ground_tform_idx] += reduced_tform_cls_loss.item()
                    disp_tforms_correct_insts_t1[phase, stratum_ground_tform_idx] += reduced_correct_insts_t1.item()
                    disp_tforms_correct_insts_t5[phase, stratum_ground_tform_idx] += reduced_correct_insts_t5.item()
                    disp_tforms_total_insts[phase, stratum_ground_tform_idx] += len(gtform_stratum_dindices)

                # phase_name == 'train' and not eval_only => update xlayer
                if not eval_only and phase_name == 'train':
                    optimizer.zero_grad()
                    with amp.scale_loss(cls_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    optimizer.step()

                    adjust_learning_rate(optimizer, epoch + frac_epoch)

                end = time.time()
                batch_time = end - start
                batch_time_stat.update(phase, batch_time)
                start = time.time()
            # end of phase loop
        # end of input tform loop
        disp_interval += 1
        
        if (step+1) % args.disp_interval == 0 or (step+1) == iters_per_epoch:
            disp_tforms_cls_loss    /= disp_interval
            disp_tforms_correct_acc_t1 = disp_tforms_correct_insts_t1 / disp_tforms_total_insts
            disp_tforms_correct_acc_t5 = disp_tforms_correct_insts_t5 / disp_tforms_total_insts
            
            for phase in range(2):
                phase_name = phase_names[phase]
                imgs_per_sec = args.world_size * len(im_data0) / batch_time_stat.avg[phase]
                for input_tform_idx in range(num_tforms):
                    tform_name = lenses_spec[input_tform_idx][0]
                    print0("%3d/%3d t %.3f s %d, %s %s cl: %.2f, acc: %.3f/%.3f" \
                           % (step+1, iters_per_epoch, batch_time_stat.avg[phase],
                              imgs_per_sec, phase_name, tform_name,
                              disp_tforms_cls_loss[phase, input_tform_idx],
                              disp_tforms_correct_acc_t1[phase, input_tform_idx],
                              disp_tforms_correct_acc_t5[phase, input_tform_idx]))
                                                                        
            disp_tforms_avg_acc_diff_t1 = disp_tforms_correct_acc_t1[1] - disp_tforms_correct_acc_t1[0]
            disp_tforms_avg_acc_diff_t5 = disp_tforms_correct_acc_t5[1] - disp_tforms_correct_acc_t5[0]

            disp_tforms_cls_loss[...] = 0
            disp_interval = 0
            batch_time_stat.reset()

            tforms_all_correct_insts_t1 += disp_tforms_correct_insts_t1
            tforms_all_correct_insts_t5 += disp_tforms_correct_insts_t5
            tforms_total_insts += disp_tforms_total_insts - DELTA
            tforms_all_avg_acc_diff_t1 = ( tforms_all_correct_insts_t1[1] - tforms_all_correct_insts_t1[0] ) / tforms_total_insts[0]
            tforms_all_avg_acc_diff_t5 = ( tforms_all_correct_insts_t5[1] - tforms_all_correct_insts_t5[0] ) / tforms_total_insts[0]

            for input_tform_idx in range(1, num_tforms):
                print0("%-14s %d t1/t5 diff: %.1f/%.1f, overall t1/t5 diff: %.1f/%.1f" %\
                        (lenses_spec[input_tform_idx][0], step+1,
                         disp_tforms_avg_acc_diff_t1[input_tform_idx]*100, disp_tforms_avg_acc_diff_t5[input_tform_idx]*100,
                         tforms_all_avg_acc_diff_t1[input_tform_idx]*100,  tforms_all_avg_acc_diff_t5[input_tform_idx]*100))

                if args.local_rank == 0:
                    board.add_scalar('%d-t1-diff' %(input_tform_idx),  \
                                        disp_tforms_avg_acc_diff_t1[input_tform_idx]*100, step+1)
                    board.add_scalar('%d-t5-diff' %(input_tform_idx),  \
                                        disp_tforms_avg_acc_diff_t5[input_tform_idx]*100, step+1)

            disp_tforms_correct_insts_t1[...] = 0
            disp_tforms_correct_insts_t5[...] = 0
            disp_tforms_total_insts[...] = DELTA
            start = time.time()

        # remember best prec@1 and save checkpoint
        if (not eval_only) and args.local_rank == 0 \
               and ((step+1) % args.save_model_interval == 0 or step+1 == iters_per_epoch):
            cp_filepath = "%s/%s-%s-%d-%04d.pth" %(args.cp_dir, args.architecture, args.host_stage_name, epoch, step+1)
            save_checkpoint({
                    'epoch':      epoch,
                    'arch':       args.architecture,
                    'state_dict': xlayer.state_dict(),
                    'opt_state' : optimizer.state_dict(),
                    'opt_name':   args.optimizer,
                    'world_size': args.world_size
                }, filename=cp_filepath, model=xlayer)

    for input_tform_idx in range(num_tforms):
        avg_acc_t1 = tforms_all_correct_insts_t1[:, input_tform_idx] / tforms_total_insts[0, input_tform_idx]
        avg_acc_t5 = tforms_all_correct_insts_t5[:, input_tform_idx] / tforms_total_insts[0, input_tform_idx]
        print0("%-14s overall orig t1/t5: %.1f/%.1f, xlayer t1/t5: %.1f/%.1f" %\
                (lenses_spec[input_tform_idx][0],
                 avg_acc_t1[0]*100, avg_acc_t5[0]*100,
                 avg_acc_t1[1]*100, avg_acc_t5[1]*100))

def main():
    if not args.amp:
        args.opt_level = 'O0'
        print0("fp32 only")

    print0("opt_level = {}".format(args.opt_level))
    print0("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32))
    print0("loss_scale = {}".format(args.loss_scale))

    cudnn.benchmark = True
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)

    # Set the featlens module's global variable CALC_ORIGFEAT_PEARSON to control the output stats
    # featlens.CALC_ORIGFEAT_PEARSON default: False
    featlens.CALC_ORIGFEAT_PEARSON = args.calc_origfeat_pearson
    featlens.DEBUG = args.debug
    featlens.USING_MAE = args.using_mae_for_whole_act_loss
    
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.distributed = args.world_size > 1
    else:
        args.gpu = 0
        args.world_size = 1

    args.dataset0 = args.dataset
    # google open images
    if args.dataset == 'open':
        args.dataset = 'imagenet'
        
    default_optimizer = { 'imagenet': 'sgd', 'mnist': 'ranger', 'cifar10': 'ranger' }
    default_lr = { 'imagenet': {'sgd': 0.01, 'ranger': 0.005, 'adam': 0.01},
                   'mnist':    {'sgd': 0.01, 'ranger': 0.01, 'adam': 0.01},
                   'cifar10':    {'sgd': 0.01, 'ranger': 0.01, 'adam': 0.01} }

    default_gamma = { 'imagenet': {'sgd': 0.5, 'ranger': 0.5, 'adam': 0.5},
                      'mnist':    {'sgd': 0.5, 'ranger': 0.5, 'adam': 0.5},
                      'cifar10':  {'sgd': 0.5, 'ranger': 0.5, 'adam': 0.5} }
                        
    default_weight_decay = { 'sgd': 0.01, 'ranger': 0.1, 'adam': 0.01 }
    # topk of each feature map (should be multiplied with batch size)
    default_base_act_topk = { 'imagenet-res4': 6, 'imagenet-res3': 12, 
                              'mnist-res4': 3,    'mnist-res3': 6,
                              'cifar10-res4': 3,  'cifar10-res3': 6 }
                                
    default_rot_cls_lr_scale    = { 'imagenet': 0.05, 'mnist': 1.0, 'cifar10': 1.0 }
    default_total_epochs        = { 'imagenet': 2,    'mnist': 6,   'cifar10': 6, 'open': 2 }
    default_lr_decay_epoch_step = { 'imagenet': 0.25, 'mnist': 1.0, 'cifar10': 1.0,
                                    'open': 0.25 }
        
    if args.optimizer is None:
        args.optimizer = default_optimizer[args.dataset]
    if args.lr == -1:
        args.lr = default_lr[args.dataset][args.optimizer]
        if args.using_xlayer_resnet:
            args.lr *= 2
        
    if args.gamma == -1:
        args.gamma = default_gamma[args.dataset][args.optimizer]
    if args.weight_decay == -1:
        args.weight_decay = default_weight_decay[args.optimizer]
    if args.total_epochs == -1:
        args.total_epochs = default_total_epochs[args.dataset0]
        if args.using_xlayer_resnet:
            # 4 epochs on imagenet, 8 epochs on mnist
            args.total_epochs += 2
            
    args.lr_decay_epoch_step = default_lr_decay_epoch_step[args.dataset0]
    if args.using_xlayer_resnet and args.dataset == 'imagenet':
        args.lr_decay_epoch_step *= 2

    args.dataset_hoststage = "%s-%s" %(args.dataset, args.host_stage_name)
    args.base_act_topk = default_base_act_topk[args.dataset_hoststage]
        
    if args.dataset == 'mnist' or args.dataset == 'cifar10':
        args.train_batch_size = 128
        args.normal_batch_size = 128
        args.test_batch_size = 1000
        args.do_pool1 = False
        args.num_classes = 10
        args.use_pretrained = False
        args.architecture = 'resnet18'
        args.rot_feat_shape = (1024, 4, 4)
        args.rot_cls_avgpool_size = None # use all features for rotation classification
        if args.dataset == 'mnist':
            args.host_cp_filepath = 'resnet18-mnist-1206-9.pth'
        if args.dataset == 'cifar10':
            args.host_cp_filepath = 'resnet18-cifar10-0305-59.pth'
    else:
        args.normal_batch_size = 256
        args.do_pool1 = True
        args.num_classes = 1000
        args.use_pretrained = True
        if args.using_xlayer_resnet:
            args.test_batch_size = args.train_batch_size
        else:
            args.test_batch_size = int(args.train_batch_size * 1.5)
        
        if args.host_stage_name == 'res4':
            args.rot_feat_shape = (2560, 7, 7)  # 512 conv2 + 2048 shortcut
            args.rot_cls_avgpool_size = None 
        elif args.host_stage_name == 'res3':
            args.rot_feat_shape = (1280, 14, 14)  # 256 conv2 + 1024 shortcut
            args.rot_cls_avgpool_size = (7,7) # pool 14*14 to 7*7 to reduce RAM and overfitting
            args.lr *= 5                     # res3 requires larger lr
        else:
            raise NotImplementedError

    args.train_batch_size //= args.world_size
    args.test_batch_size  //= args.world_size

    # Scale learning rate based on global batch size. lr increases
    args.lr = args.lr * float(args.train_batch_size * args.world_size) / args.normal_batch_size
    
    datehour = datetime.datetime.now().strftime("%m%d%H")
    args.cp_dir = os.path.join(os.environ['IMAGENET'], 'featlens',
                           '%s-%s%.4g-%s' %(args.dataset0, args.optimizer, args.lr, datehour))
                                                  
    n_gpu = torch.cuda.device_count()

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    print0("n_gpu: {}, world size: {}, rank: {}, amp: {}".format(
                n_gpu, args.world_size, args.local_rank, args.amp))
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    arch_match = re.match("^(resnet)", args.architecture)
    if not arch_match:
        raise NotImplementedError
    arch_type = arch_match.group(1)

    # enable_lenses default is True.
    # if args.enable_lenses = False, the original pretrained model will be evaluated
    # should yield the same performance as when lenskit.using_lenses = False
    if (args.eval_only and args.lenses_cp_filepath is None):
        args.enable_lenses = False
        
    if args.using_xlayer_resnet:
        args.enable_lenses = False
        args.host_stage_name = "xlayer"
        
    if args.architecture == 'resnet101':
        net = resnet101(pretrained=args.use_pretrained,
                        lens_stage=args.host_stage_name,
                        num_classes=args.num_classes,
                        do_pool1=args.do_pool1)
    elif args.architecture == 'resnet50':
        net = resnet50(pretrained=args.use_pretrained,
                       lens_stage=args.host_stage_name,
                       num_classes=args.num_classes,
                       do_pool1=args.do_pool1)
    elif args.architecture == 'resnet34':
        net = resnet34(pretrained=args.use_pretrained,
                       lens_stage=args.host_stage_name,
                       num_classes=args.num_classes,
                       do_pool1=args.do_pool1)
    elif args.architecture == 'resnet18':
        net = resnet18(pretrained=args.use_pretrained,
                       lens_stage=args.host_stage_name,
                       num_classes=args.num_classes,
                       do_pool1=args.do_pool1)
    else:
        raise NotImplementedError

    if args.host_cp_filepath:
        load_host_checkpoint(net, args.host_cp_filepath)

    # in typical use, we freeze BNs in pretrained models
    # so for fair comparison, we freeze them here as well
    if args.update_all_batchnorms:
        net.train()
    else:
        net.eval()

    default_lenses_set = { 'imagenet': {'rot': 1, 'scaling': 1}, 
                           'mnist':    {'rot': 1, 'scaling': 0},
                           'cifar10':  {'rot': 1, 'scaling': 0} }
                            
    args.using_rot_lenses     = args.using_rot_lenses and \
                                default_lenses_set[args.dataset]['rot']
    args.using_scaling_lenses = args.using_scaling_lenses and \
                                default_lenses_set[args.dataset]['scaling']
    if args.using_rot_lenses:
        num_rot_lenses = len(rot_lenses_spec)
    else:
        num_rot_lenses = 0
    if args.using_scaling_lenses:
        num_scaling_lenses = len(scaling_origsize_lenses_spec)
    else:
        num_scaling_lenses = 0
        
    print0("%d rotation lenses, %d scaling lenses" %(num_rot_lenses, num_scaling_lenses))
    # rot_lenses_only: only train/test rotation lenses. 
    # In this case, we do different random rotations to images in each batch.
    args.rot_lenses_only = args.using_rot_lenses and not args.using_scaling_lenses
        
    # in this simplified setting, only guess rotation lenses 
    # if there are only rotation lenses in the lenskit
    if not args.rot_lenses_only:
        args.guessing_rot_lenses = False
        args.rot_feat_shape = None
    
    global rot_group_range2index, rot_group_ranges, num_rot_groups
    rot_group_ranges0 = [0, 45, 135, 225, 315, 360]
    rot_group_range2index = torch.tensor([0, 1, 2, 3, 0]).cuda()
    num_rot_groups = torch.unique(rot_group_range2index).numel()
    rot_group_ranges = []
    for i in range(len(rot_group_ranges0) - 1):
        rot_group_ranges.append( [rot_group_ranges0[i], rot_group_ranges0[i+1]] )
    rot_group_ranges = torch.tensor(rot_group_ranges).cuda()
    # rot_group_ranges: 1000*5*2 for mnist. The repeated dim should be big enough
    rot_group_ranges = rot_group_ranges.unsqueeze(0).repeat(args.test_batch_size, 1, 1)
    lenses_spec = list(orig_lens_spec) # lenses_spec has only 1 lens now
    if args.using_rot_lenses:
        lenses_spec += rot_lenses_spec  # rot lenses always have indices 1~3
    if args.using_scaling_lenses:
        # default do_upsampling=True, which performs slightly better
        if args.do_upsampling:
            lenses_spec += scaling_upsample_lenses_spec
        else:
            lenses_spec += scaling_origsize_lenses_spec

    for lens_spec in lenses_spec:
        # the transformation modules are put in a list in the lenskit,
        # so need to manually cuda() here
        for attr in lens_spec:
            if isinstance(attr, nn.Module):
                attr.cuda()
        if type(lens_spec[3]) == list and not args.interp_origfeat:
            # disable adding orig_feat in MixConv2L
            lens_spec[3][4] = None

    # Only consider res3 and res4. both discounts are 0.2
    resnet_overshoot_discounts = [-1, -1, -1,  0.2,  0.2]

    if not args.using_xlayer_resnet:
        # created lenskit is registered as net.lenskit
        create_lenskit(net, arch_type, args.host_stage_name, 
                       lenses_spec, resnet_overshoot_discounts)

    if args.sync_bn:
        import apex
        print0("using apex synced BN")
        net = apex.parallel.convert_syncbn_model(net)

    net = net.cuda()
    # net.lenskit is on cuda now
    if args.using_xlayer_resnet:
        # xlayer is the last BasicBlock/Bottleneck in net.layer4
        plugin = net.layer4[-1]
    else:
        plugin = net.lenskit
        
    plugin_params = list(plugin.named_parameters())
    plugin_param_names = [ n for n,p in plugin_params ]
    # param_to_moduleName is used in init_optimizer() to filter out
    # bias and BN terms for different weight decays
    param_to_moduleName = {}
    for m in plugin.modules():
        for p in m.parameters(recurse=False):
            param_to_moduleName[p] = type(m).__name__
    
    if args.dataset0 == 'imagenet':
        train_loader, val_loader = create_imagenet_data_loader()
    elif args.dataset0 == 'mnist':
        train_loader, val_loader = create_mnist_data_loader(normalizing_mean_std=True,
                                        using_mnist_rot=True, out_rot_angle=True,
                                        out_orig_image=True)
    elif args.dataset0 == 'cifar10':
        train_loader, val_loader = create_cifar10_data_loader()
    elif args.dataset0 == 'open':
        # open images have no validation set (no class label)
        # use imagenet val set for validation
        train_loader, _ = create_open_data_loader()
        _, val_loader = create_imagenet_data_loader()
        
    if not args.eval_only:
        num_train_opt_steps = int( len(train_loader) / args.train_batch_size ) * args.total_epochs
        optimizer = init_optimizer(plugin_params, param_to_moduleName,
                                   num_train_opt_steps, 
                                   default_rot_cls_lr_scale[args.dataset])
        print0("Optimizer =", optimizer)
        print0("Initial LR: %.4f" %args.lr)
        print0("LR will be decayed by %.3f every %.3f epochs" %(args.gamma, args.lr_decay_epoch_step))
        # only use amp during training
        real_model, net, optimizer = init_amp(net, optimizer)
        if not os.path.isdir(args.cp_dir) and args.local_rank == 0:
            os.makedirs(args.cp_dir)
        
    else:
        real_model = net
        optimizer = None

    if args.lenses_cp_filepath is not None:
        resumed_epoch, finished_iters, cp_world_size = load_plugin_checkpoint(plugin, optimizer)
    else:
        resumed_epoch, finished_iters, cp_world_size = 0, 0, 1

    if args.using_xlayer_resnet:
        train_eval_func = train_eval_resnet
    else:
        train_eval_func = train_eval_lenses
    
    # plugin = xlayer if args.using_xlayer_resnet
    # plugin = lenskit otherwise
    if args.eval_only:
        # epoch, resumed_epoch, finished_iters, cp_world_size
        # cp_world_size * finished_iters = actually finished iters without parallelism
        train_eval_func(0, 0, 0, 1,
                   net, real_model, val_loader, None, 
                   plugin, lenses_spec, eval_only=True,
                   enable_lenses=args.enable_lenses)
        return

    global old_decay_count
    old_decay_count = 0
    
    for epoch in range(resumed_epoch, args.total_epochs):
        train_eval_func(epoch, resumed_epoch, finished_iters, cp_world_size,
                   net, real_model, train_loader, optimizer, 
                   plugin, lenses_spec, eval_only=False,
                   enable_lenses=args.enable_lenses)
        train_eval_func(epoch, 0,             0,              cp_world_size,
                   net, real_model, val_loader, None,
                   plugin, lenses_spec, eval_only=True,
                   enable_lenses=args.enable_lenses)

if __name__ == '__main__':
    main()
