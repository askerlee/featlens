import argparse
import os
import shutil
import time
from datetime import date
import re

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch._six import inf
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import reflectnet
import resnet
from ranger import Ranger
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from tensorpack import imgaug

import numpy as np
from lmdbloader import LMDBLoader, fbresnet_augmentor
import socket
hostname = socket.gethostname()
import pdb

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

lab_hostnames = [ 'desktop', 'workstation', 'workstation2' ]
if hostname == 'm10':
    os.environ['IMAGENET'] = '/media/xxxxxx/ssd'
elif hostname in lab_hostnames:
    os.environ['IMAGENET'] = '/data/xxxxxx'
else:
    print("Unknown hostname '%s'. Please specify 'IMAGENET' manually." %hostname)
    exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='reflectnet18',
                        help='model architecture (default: resnet18)')
    parser.add_argument('--cycles', default=3, type=int, metavar='N',
                        help='number of model reflection cycles (default: 2)')
    parser.add_argument('--cycletrans', dest='cycle_trans', 
                        default=None, type=str, 
                        choices=[None, 'conv1x1', 'roll_channels'], 
                        help='Transition layer between cycles (default: roll_channels)')
    parser.add_argument('--runcycles', default=-1, type=int, metavar='N',
                        help='number of runtime reflection cycles (default: same as --cycles)')
    parser.add_argument('--start-epoch', dest='start_epoch', default=-1, type=int, metavar='N',
                        help='Manual start epoch (default: if resume_cp, N=cp_epoch. otherwise N=0)')
    parser.add_argument('--freezebone', action='store_true',
                        help='Freeze backbone feature extractor')
    parser.add_argument('--dataset', dest='dataset',
                        help='The type of dataset (default: imagenet)',
                        default='imagenet', type=str,
                        choices=['imagenet', 'tiny-imagenet', 'mnist', 'mnist-rot'])
    parser.add_argument('-j', dest='num_workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--bs', dest='train_batch_size', default=256, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--lr', default=-1, type=float,
                        metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.train_batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--decay', dest='weight_decay', default=-1, type=float,
                        metavar='W', help='weight decay (default: 0.1 for reflectnet, 0.001 for resnet)')
    parser.add_argument('--gamma', default=-1, type=float,
                        metavar='Y', help='learning rate decay factor (default: 0.1 for SGD and 0.2 for others)')
    parser.add_argument('--clip', default=2, type=float,
                        metavar='C', help='gradient clip to C (default: 2)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--cp', dest='resume_cp', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--eval', dest='eval_only', action='store_true',
                        help='only evaluate model on validation set')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--opt', dest='optimizer',
                      help='The type of optimizer',
                      default=None, type=str)
    parser.add_argument('--geo', dest='do_geo_aug', action='store_true')
    parser.add_argument('--testgeo', dest='test_geo_trans_index', 
                      default=0, type=int)
    parser.add_argument('--in', dest='input_tensor',
                      help='Input tensor checkpoint filename',
                      default=None, type=str)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')

    parser.add_argument('--opt-level', dest='opt_level', default='O2', type=str)
    parser.add_argument('--keep-batchnorm-fp32', dest='keep_batchnorm_fp32', type=str, default=None)
    parser.add_argument('--loss-scale', dest='loss_scale', type=str, default=None)
    parser.add_argument('--no-async-prefetcher', dest='async_prefetcher', action='store_false')
    parser.add_argument('--notestbn', dest='no_testbn',
                      help='Do not set eval() mode during test, instead use instance normalization',
                      action='store_true')
    parser.add_argument('--tta', help='Do test time augmentation', action='store_true')
    parser.add_argument('--noamp', dest='amp', help='Do not use mixed precision', 
                        action='store_false')
    
    args = parser.parse_args()
    return args

def print0(*print_args, **kwargs):
    if args.local_rank == 0:
        print(*print_args, **kwargs)

def set_cycles(num_cycles):
    def real_set_cycles(m):
        classname = m.__class__.__name__
        if classname == 'BasicBlock' or classname == 'Bottleneck':
            m.num_cycles = num_cycles
    
    return real_set_cycles

     
def main():
    global best_prec1, args

    args = parse_args()

    if not args.amp:
        args.opt_level = 'O0'
        print0("fp32 only")
            
    print0("opt_level = {}".format(args.opt_level))
    print0("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32))
    print0("loss_scale = {}".format(args.loss_scale))

    # on MNIST, adam performs better than ranger
    default_optimizer = { 'imagenet': 'sgd', 'tiny-imagenet': 'sgd', 
                          'mnist': 'adam',   'mnist-rot': 'ranger' }
    default_lr = { 'imagenet': {'sgd': 0.1, 'ranger': 0.001, 'adam': 0.001},
                   'mnist':    {'sgd': 0.001, 'ranger': 0.0001, 'adam': 0.0001} }
                    
    default_gamma = { 'imagenet': {'sgd': 0.1, 'ranger': 0.2, 'adam': 0.2},
                      'mnist':    {'sgd': 0.4, 'ranger': 0.4, 'adam': 0.4} }
    default_gamma_epoch = { 'imagenet': 20, 'mnist': 3 }                    
    default_weight_decay = {   'sgd': {'reflectnet': 0.001, 'resnet': 0.0001},
                            'ranger': {'reflectnet': 0.1, 'resnet': 0.01},
                            'adam':   {'reflectnet': 0.1, 'resnet': 0.01} }
    
    if args.dataset == 'mnist-rot':
        args.dataset2 = 'mnist'
    elif args.dataset == 'tiny-imagenet':
        args.dataset2 = 'imagenet'
    else:
        args.dataset2 = args.dataset
        
    if args.optimizer is None:
        args.optimizer = default_optimizer[args.dataset]
            
    if args.lr == -1:
        args.lr = default_lr[args.dataset2][args.optimizer]
    if args.gamma == -1:
        args.gamma = default_gamma[args.dataset2][args.optimizer]
    args.gamma_epoch = default_gamma_epoch[args.dataset2]
        
    cudnn.benchmark = True
    best_prec1 = 0
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.distributed = args.world_size > 1
    else:
        args.gpu = 0
        args.world_size = 1

    args.train_batch_size //= args.world_size
    args.test_batch_size = int(args.train_batch_size * 1.6)

    use_reflectnet = False
    use_resnet = False
    if re.match('^reflect', args.arch):
        use_reflectnet = True
    elif re.match('^resnet', args.arch):
        use_resnet = True
    else:
        print0("Only reflectnet and resnet are supported")
        exit(0)

    if args.weight_decay == -1:
        if use_resnet:
            args.weight_decay = default_weight_decay[args.optimizer]['resnet']
        if use_reflectnet:
            args.weight_decay = default_weight_decay[args.optimizer]['reflectnet']
    
    if args.dataset2 == 'mnist':
        args.train_batch_size = 128
        args.test_batch_size = 1000
        args.do_pool1 = False
        args.num_classes = 10
        args.warmup_epochs = 0
        args.epochs = 10
        args.normal_batch_size = args.train_batch_size
    elif args.dataset2 == 'imagenet':
        args.do_pool1 = True
        args.normal_batch_size = 256.
        if args.dataset == 'imagenet':
            args.num_classes = 1000
            args.warmup_epochs = 5
        else:
            args.num_classes = 200
            args.warmup_epochs = 2
            
    print0("Weight decay: %.4f. LR decay: %.4f every %d epochs" %( \
                args.weight_decay, args.gamma, args.gamma_epoch))
                
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
        
    # create model
    if args.pretrained:
        print0("=> using pre-trained model '{}', {} classes".format(args.arch, args.num_classes))
        if use_resnet:
            model = resnet.__dict__[args.arch](pretrained=True, 
                                               num_classes=args.num_classes, 
                                               do_pool1=args.do_pool1)
    else:
        if use_resnet:
            model = resnet.__dict__[args.arch](num_classes=args.num_classes, 
                                               do_pool1=args.do_pool1)
            print0("=> creating model '{}', {} classes".format(args.arch, args.num_classes))
        elif use_reflectnet:
            model = reflectnet.__dict__[args.arch](num_cycles=args.cycles, 
                                                   num_classes=args.num_classes,
                                                   cycle_trans=args.cycle_trans)
                                                   
            print0("=> creating model '{}', cycles: {}, {} classes".format(args.arch, 
                            args.cycles, args.num_classes))

    if use_reflectnet:
        if args.runcycles == -1:
            args.runcycles = args.cycles
        if args.runcycles != args.cycles:
            print0("Runtime cycles %d != model cycles %d" %(args.runcycles, args.cycles))
            print0("Set all BasicBlock/Bottleneck num_cycles to %d" %args.runcycles)
            model.apply(set_cycles(args.runcycles))
        
    if args.sync_bn:
        import apex
        print0("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model = model.cuda()

    # Scale learning rate based on global batch size
    args.lr = args.lr * float(args.train_batch_size * args.world_size) / args.normal_batch_size 
    if args.freezebone:
        grad_names, nograd_names = {}, {}
        for name, value in model.named_parameters():
            if re.match(r"^fc\.", name):
                grad_names[name] = 1
                continue
            value.requires_grad = False
            nograd_names[name] = 1
        
    sgd_optimizer    = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                       momentum=args.momentum, 
                                       weight_decay=args.weight_decay)
    adam_optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr, 
                                       weight_decay=args.weight_decay)
    ranger_optimizer = Ranger(model.parameters(), lr=args.lr, 
                              weight_decay=args.weight_decay)

    if args.optimizer == 'sgd':
        optimizer = sgd_optimizer
    elif args.optimizer == 'adam':
        optimizer = adam_optimizer
    elif args.optimizer == 'ranger':
        optimizer = ranger_optimizer
        
    if not args.eval_only:
        print0("Optimizer =", optimizer)
        print0("Initial LR: %.4f" %args.lr)
    
    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with 
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)
        real_model = model.module
    else:
        real_model = model
        
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # Optionally resume from a checkpoint
    if args.resume_cp:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume_cp):
                print0("=> loading checkpoint '{}'... ".format(args.resume_cp))
                checkpoint = torch.load(args.resume_cp, map_location = lambda storage, loc: storage.cuda(args.gpu))
                if args.start_epoch == -1:
                    args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                cp_num_classes = checkpoint['state_dict']['fc.weight'].shape[0]
                if cp_num_classes != args.num_classes:
                    del checkpoint['state_dict']['fc.weight']
                    del checkpoint['state_dict']['fc.bias']
                    print0("checkpoint classes %d != current classes %d, drop FC" %( \
                                cp_num_classes, args.num_classes))
                                
                real_model.load_state_dict(checkpoint['state_dict'], strict=False)
                
                if checkpoint['opt_name'] == args.optimizer:
                    optimizer.load_state_dict(checkpoint['opt_state'])
                else:
                    print0("cp opt '%s' != current opt '%s'. Skip loading optimizier states" % \
                            (checkpoint['opt_name'], args.optimizer))
                            
                print0("loaded.")
            else:
                print0("!!! no checkpoint found at '{}'".format(args.resume_cp))
                exit(0)
                
        resume()
        
    else:
        if args.start_epoch == -1:
            args.start_epoch = 0
                    
    if args.dataset == 'imagenet':
        train_loader, val_loader = create_imagenet_data_loader()
    elif args.dataset == 'tiny-imagenet':
        train_loader, val_loader = create_tiny_imagenet_data_loader()
    elif args.dataset == 'mnist':
        train_loader, val_loader = create_mnist_data_loader(normalize_mean_std=False, 
                                        use_mnist_rot=False)
    elif args.dataset == 'mnist-rot':
        train_loader, val_loader = create_mnist_data_loader(normalize_mean_std=False, 
                                        use_mnist_rot=True)
    
    # for debugging input that causes nan only                            
    if args.input_tensor:
        input_tensor = torch.load(args.input_tensor)
        model.eval()
        output = model(input_tensor)
                          
    if args.eval_only:
        cp_filename = os.path.basename(args.resume_cp)
        validate(val_loader, model, criterion, cp_filename)
        return

    args.cp_sig = args.dataset
    if args.do_geo_aug:
        args.cp_sig += '-geo'
    
    today = date.today()
    today_str = today.strftime("%m%d")
    args.save_dir = os.path.join(os.environ['IMAGENET'], 'reflectnet', 
                                 "%s-%s-%s" %(args.arch, args.cp_sig, today_str))
                                 
    if args.local_rank == 0:
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        print("Models will be saved to '%s'" %args.save_dir)
        global board
        # start a new board
        board = SummaryWriter()

    for epoch in range(args.start_epoch, args.epochs):
        #if args.distributed:
        #    train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, "Epoch %d"%epoch, epoch)

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': real_model.state_dict(),
                'best_prec1': best_prec1,
                'opt_state' : optimizer.state_dict(),
                'opt_name': args.optimizer
            }, is_best, filename='%d.pth' %epoch)

            maxweight(real_model)
                    
# no_blocking: do non_blocking cuda operations 
# in practice it may increase loading speed slightly
# imagenet mean: [0.485, 0.456, 0.406]
# imagenet std:  [0.229, 0.224, 0.225]
class data_prefetcher():
    def __init__(self, loader, mean, std, no_blocking=False):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor(mean).cuda().view(1,3,1,1)
        self.std  = torch.tensor(std).cuda().view(1,3,1,1)
        self.no_blocking = no_blocking
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
            
        if not self.no_blocking:
            self.next_input = self.next_input.cuda()
            self.next_target = self.next_target.cuda()
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            return

        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        if not self.no_blocking:
            input_data = self.next_input
            target = self.next_target
            self.preload()
            return input_data, target
            
        torch.cuda.current_stream().wait_stream(self.stream)
        input_data = self.next_input
        target = self.next_target
        if input_data is not None:
            input_data.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input_data, target

def create_imagenet_data_loader():
    if not args.distributed:    
        train_loader = LMDBLoader('train', do_aug=True, 
                                batch_size=args.train_batch_size, shuffle=False,
                                num_workers=args.num_workers, cuda=False, 
                                out_tensor=True, data_transforms=None)
        val_loader   = LMDBLoader('val',   do_aug=args.tta, 
                                batch_size=args.test_batch_size, shuffle=False,
                                num_workers=args.num_workers, cuda=False, 
                                out_tensor=True, data_transforms=None)
    else:
        train_loader = LMDBLoader('train-%d' %args.local_rank, do_aug=True, 
                                batch_size=args.train_batch_size, shuffle=False,
                                num_workers=args.num_workers, cuda=False, 
                                out_tensor=True, data_transforms=None)
        val_loader   = LMDBLoader('val-%d' %args.local_rank,   do_aug=args.tta, 
                                batch_size=args.test_batch_size, shuffle=False,
                                num_workers=args.num_workers, cuda=False, 
                                out_tensor=True, data_transforms=None)

    return train_loader, val_loader

def imgaug_wrapper(imgaug_list):
    imgaug_augmentor = imgaug.AugmentorList(imgaug_list)
    # don't normalize the image yet. will normalize within data_prefetcher
    # don't use transforms.ToTensor(). It will divide pixels by 256
    ToTensor = torch.from_numpy
    
    # img_obj is a PIL Image object. Return a CPU tensor
    def real_augmentor(img_obj):
        img_np = np.asarray(img_obj)
        transforms = imgaug_augmentor.get_transform(img_np)
        img_np2 = transforms.apply_image(img_np)
        img_np2 = np.rollaxis(img_np2, 2)
        img_ts  = ToTensor(np.ascontiguousarray(img_np2))
        return img_ts
    
    return real_augmentor
    
def create_tiny_imagenet_data_loader():
    train_trans = imgaug_wrapper(fbresnet_augmentor(True))
    val_trans   = imgaug_wrapper(fbresnet_augmentor(False))
    
    train_dataset = datasets.ImageFolder("./nips18-avc/tiny-imagenet-200/train/", transform=train_trans)
    val_dataset   = datasets.ImageFolder("./nips18-avc/tiny-imagenet-200/val/",   transform=val_trans)
    
    train_sampler = DistributedSampler(train_dataset) \
                        if args.distributed else RandomSampler(train_dataset)
    val_sampler   = DistributedSampler(val_dataset) \
                        if args.distributed else None
    
    train_loader = DataLoader( train_dataset, sampler=train_sampler,
                               batch_size=args.train_batch_size, num_workers=args.num_workers )
    val_loader   = DataLoader( val_dataset,   sampler=val_sampler,
                               batch_size=args.test_batch_size,  num_workers=args.num_workers )
    
    return train_loader, val_loader

# overload MNIST to load the mnist-rot dataset
# when is_rot=True, orig_root specifies the directory of original MNIST images
class MNIST2(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, is_rot=False, out_rot_angle=False,
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
            
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        
        if is_rot:
            self.data, self.targets, self.rot_angles, self.orig_indices = torch.load(os.path.join(self.processed_folder, data_file))
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
            rot_angle   = self.rot_angles[index]
            orig_index = self.orig_indices[index]
            if self.out_orig_image:
                assert self.orig_targets[orig_index] == target
                orig_image = Image.fromarray(self.orig_data[orig_index].numpy(), mode='L')
                if self.transform is not None:
                    orig_image = self.transform(orig_image)
                return image, target, rot_angle, orig_image            
            else:
                return image, target, rot_angle
        else:
            return super(MNIST2, self).__getitem__(index)
 
def create_mnist_data_loader(normalize_mean_std, use_mnist_rot=False, 
                             out_rot_angle=False, out_orig_image=False):
    transforms_list = [
                      transforms.Resize((56, 56)),
                      # repeat the single channel three times to make it "RGB"
                      transforms.Grayscale(3),
                      transforms.ToTensor(),
                      ]
    
    if normalize_mean_std:
        transforms_list.append( transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)) )
                      
    normalize = transforms.Compose(transforms_list)
    
    rot_root = "../data-rot"
    orig_root  = "../data"
    if use_mnist_rot:
        mnist_data_root = rot_root
    else:
        mnist_data_root = orig_root
        
    # 10000 images for training, so use the original test set as the training set
    # pin_memory=True to put data on GPU
    # small data. no need to use multiple workers
    train_loader = DataLoader(
        MNIST2(mnist_data_root, train=True, download=not use_mnist_rot, transform=normalize, 
               is_rot=use_mnist_rot, out_rot_angle=out_rot_angle,
               orig_root=orig_root, out_orig_image=True),
        batch_size=args.train_batch_size, shuffle=False)
    # 50000 images for test,     so use the original training set as the test set
    val_loader = torch.utils.data.DataLoader(
        MNIST2(mnist_data_root, train=False, transform=normalize, 
               is_rot=use_mnist_rot, out_rot_angle=out_rot_angle,
               orig_root=orig_root, out_orig_image=True),
        batch_size=args.test_batch_size,  shuffle=False)
    
    return train_loader, val_loader                                      
                 
def maxweight(model):
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
    
    print0( "max '%s' %.4f, min '%s' %.4f" %( \
            max_param_name, weight_max, min_param_name, weight_min) )

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

all_geo_trans = [ identity(), RotateTensor4d(90), RotateTensor4d(180), RotateTensor4d(270), 
                  ScaleTensor4d(0.5, 0.5), ScaleTensor4d(0.33, 0.33) ]
geo_trans_prob = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]

def rand_geo_trans(img_tensor):
    geo_trans_idx = np.random.choice(len(geo_trans_prob), p=geo_trans_prob)
    return all_geo_trans[geo_trans_idx](img_tensor)
    
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    if args.distributed:
        real_model = model.module
    else:
        real_model = model

    if args.dataset == 'imagenet':
        mean, std = (0.485, 0.456, 0.406),  (0.229, 0.224, 0.225)
    elif args.dataset == 'tiny-imagenet':
        mean, std = (0.466, 0.426, 0.373),  (0.282, 0.272, 0.279)
    elif args.dataset2 == 'mnist':
        mean, std = (0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)
    prefetcher = data_prefetcher(train_loader, mean, std, args.async_prefetcher)
        
    input_data, target = prefetcher.next()
    
    step = 0
    while input_data is not None:
        step += 1

        old_lr = optimizer.param_groups[0]['lr']
        new_lr = adjust_learning_rate(optimizer, epoch, step, len(train_loader), 
                                      gamma_epoch=args.gamma_epoch, warmup_epochs=args.warmup_epochs)
        if new_lr != old_lr:
            print0("LR adjust: %.5f => %.5f" %(old_lr, new_lr))
        
        if args.do_geo_aug:
            input_data2 = rand_geo_trans(input_data)
        else:
            input_data2 = input_data
            
        # compute output
        output = model(input_data2)
        loss = criterion(output, target)
        if epoch == 0 and step == 1:
            input_data2_flat = input_data2.permute(1, 0, 2, 3).reshape(3, -1)
            means =  ",".join( "%.3f" %p for p in input_data2_flat.mean(dim=1).cpu().numpy() )
            stds  =  ",".join( "%.3f" %p for p in input_data2_flat.std(dim=1).cpu().numpy() )
            print0("Data shape: %s, mean: %s, std: %s, feat: %s" % \
                            (list(input_data2.shape), means, 
                             stds, list(real_model.feat_shape)))
                                     
        # compute gradient and do SGD step
        optimizer.zero_grad()

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        if args.clip > 0:
            # `clip_grad_norm` helps prevent the exploding gradient problem 
            # caused by exploding activations
            parameters = list(filter(lambda p: p.grad is not None, amp.master_params(optimizer)))
            # 10 is usually much bigger than the actual max grad. So clip_grad_norm_() only finds 
            # the max grad, without actually doing clipping
            total_maxnorm1 = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 10, inf)
            total_l2norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip)
            total_maxnorm2 = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 10, inf)
            if args.debug:
                print0("max grad1: %.3f, total l2 norm: %.3f, max grad2: %.3f" % \
                        (total_maxnorm1, total_l2norm, total_maxnorm2))
                maxweight(real_model)
            
        # for param in model.parameters():
        #     print(param.data.double().sum().item(), param.grad.data.double().sum().item())

        optimizer.step()

        if step%args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
   
            # Average loss and accuracy across processes for logging 
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data
   
            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input_data.size(0))
            top1.update(to_python_float(prec1), input_data.size(0))
            top5.update(to_python_float(prec5), input_data.size(0))
    
            torch.cuda.synchronize()
            batch_time.update((time.time() - end)/args.print_freq)
            end = time.time()

            print0('[{0}][{1: 4d}/{2}] '
                      't {batch_time.val:.3f} '
                      's {3:.0f} '
                      'l {losses.val:.3f} ({losses.avg:.3f}) '
                      't1 {top1.val:.1f} ({top1.avg:.1f}) '
                      't5 {top5.val:.1f} ({top5.avg:.1f})'.format(
                       epoch, step, len(train_loader),
                       args.world_size*args.train_batch_size/batch_time.val,
                       batch_time=batch_time,
                       losses=losses, top1=top1, top5=top5))
                       
        input_data, target = prefetcher.next()

def validate(val_loader, model, criterion, model_str, epoch=-1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if not args.no_testbn:
        # switch to evaluate mode
        model.eval()

    end = time.time()

    if args.test_geo_trans_index > 0:
        geo_trans = all_geo_trans[args.test_geo_trans_index]
        print("Do test time geometric transformation '%s'" %(geo_trans))
    else:
        geo_trans = None

    if args.dataset == 'imagenet':
        mean, std = (0.485, 0.456, 0.406),  (0.229, 0.224, 0.225)
        mean = [mi*255 for mi in mean]
        std  = [si*255 for si in std]
    elif args.dataset == 'tiny-imagenet':
        mean, std = (0.466, 0.426, 0.373),  (0.282, 0.272, 0.279)
    elif args.dataset2 == 'mnist':
        mean, std = (0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)
    prefetcher = data_prefetcher(val_loader, mean, std, args.async_prefetcher)
        
    input_data, target = prefetcher.next()
    step = 0
    while input_data is not None:
        step += 1

        # compute output
        with torch.no_grad():
            if geo_trans:
                input_data = geo_trans(input_data)
            output = model(input_data)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data
            if torch.isnan(reduced_loss).any():
                pdb.set_trace()

        losses.update(to_python_float(reduced_loss), input_data.size(0))
        top1.update(to_python_float(prec1), input_data.size(0))
        top5.update(to_python_float(prec5), input_data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if args.local_rank == 0 and step % args.print_freq == 0:
            print('Test: [{0}/{1}] '
                  't {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  's {2:.3f} '
                  'l {losses.val:.3f} ({losses.avg:.3f}) '
                  't1 {top1.val:.1f} ({top1.avg:.1f}) '
                  't5 {top5.val:.1f} ({top5.avg:.1f})'.format(
                   step, len(val_loader),
                   args.world_size * args.test_batch_size / batch_time.val,
                   batch_time=batch_time, losses=losses,
                   top1=top1, top5=top5))


        input_data, target = prefetcher.next()

    print0('* {0}: Prec@1 {top1.avg:.1f} Prec@5 {top5.avg:.1f}'
              .format(model_str, top1=top1, top5=top5))

    if args.local_rank == 0 and epoch >= 0:
        board.add_scalar('t1-acc', top1.avg, epoch)
        board.add_scalar('t5-acc', top5.avg, epoch)

    return top1.avg


def save_checkpoint(state, is_best, filename):
    save_path = os.path.join(args.save_dir, filename)
    torch.save(state, save_path)
    print("Save model to '%s'" %save_path)
    if is_best:
        best_name = 'model_best.pth'
        best_path = os.path.join(args.save_dir, best_name)
        shutil.copyfile(save_path, best_path)
        print("Best model so far. Saved to '%s'." %best_path)

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


def adjust_learning_rate(optimizer, epoch, step, len_epoch, gamma_epoch=20, warmup_epochs=5):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    is_sgd = type(optimizer).__name__ == 'SGD'
    factor = epoch // gamma_epoch

    if is_sgd and epoch >= 80:
        factor = factor + 1

    lr = args.lr*(args.gamma**factor)

    """Warmup"""
    # if warmup_epochs is set to 0, warmup is disabled
    if is_sgd and epoch < warmup_epochs:
        lr = lr * float(1 + step + epoch*len_epoch) / (warmup_epochs * len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':
    main()
