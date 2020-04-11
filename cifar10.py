'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from ranger import Ranger

import os
import argparse
import numpy as np

from resnet import resnet101, resnet50, resnet34, resnet18
from utils import progress_bar
import pdb

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--opt', default='sgd', type=str, help='Optimizer')
parser.add_argument('--cp',  default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--100', dest='cifar100', action='store_true', help='Train on CIFAR100')
parser.add_argument('--geo', dest='do_geo_aug', action='store_true')
parser.add_argument('--testgeo', dest='test_geo_trans_index', default=0, type=int)
parser.add_argument('--eval', dest='eval_only', action='store_true',
                        help='only evaluate model on validation set')

args = parser.parse_args()

device = 'cuda'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
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

if args.cifar100:
    trainset = torchvision.datasets.CIFAR100(root='./cifar100-data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./cifar100-data', train=False, download=True, transform=transform_test)
    num_classes = 100
else:    
    trainset = torchvision.datasets.CIFAR10(root='./cifar10-data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./cifar10-data', train=False, download=True, transform=transform_test)
    num_classes = 10
    
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=0)

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = resnet18(num_classes=num_classes, do_pool1=False)
net = net.to(device)

if args.cp is not None:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.cp)
    net.load_state_dict(checkpoint['state_dict'])
    best_acc = checkpoint['best_prec1']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
sgd_optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
exp_lr_scheduler = StepLR(sgd_optimizer, step_size=10, gamma=0.4)
ranger_optimizer = Ranger(net.parameters(), lr=args.lr * 0.1, weight_decay=1e-3)
if args.opt == 'sgd':
    optimizer = sgd_optimizer
if args.opt == 'ranger':
    optimizer = ranger_optimizer

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
 
all_geo_trans =  [ identity(), RotateTensor4d(90), RotateTensor4d(180), RotateTensor4d(270) ]
geo_trans_prob = [ 0.4, 0.2, 0.2, 0.2 ]

def rand_geo_trans(img_tensor):
    geo_trans_idx = np.random.choice(len(geo_trans_prob), p=geo_trans_prob)
    return all_geo_trans[geo_trans_idx](img_tensor)
    
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # decay once for each epoch
    if args.opt == 'sgd':
        exp_lr_scheduler.step()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if args.do_geo_aug:
            inputs2 = rand_geo_trans(inputs)
        else:
            inputs2 = inputs
        
        optimizer.zero_grad()
        outputs = net(inputs2)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
    print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    if args.test_geo_trans_index > 0:
        geo_trans = all_geo_trans[args.test_geo_trans_index]
        print("Do test time geometric transformation '%s'" %(geo_trans))
    else:
        geo_trans = None
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if geo_trans:
                inputs = geo_trans(inputs)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'state_dict': net.state_dict(),
            'best_prec1': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
    
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        torch.save(state, 'checkpoint/resnet18-%s-0305-%d.pth' \
                          %('cifar100' if args.cifar100 else 'cifar10', epoch))
        best_acc = acc

if args.eval_only:
    test(0)
    exit(0)
    
for epoch in range(start_epoch, start_epoch+80):
    train(epoch)
    test(epoch)
