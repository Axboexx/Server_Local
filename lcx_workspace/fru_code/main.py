"""
@Time : 2023/7/12 17:54
@Author : Axboexx
@File : main.py
@Software: PyCharm
"""
import argparse
import os

import torch.nn as nn
import torch
from torch.backends import cudnn
from main_lib import if_mkdir
from main_lib import train
from main_lib import validate
from main_lib import merge_args_cfg
from main_lib import load_cfg
from optim import *
from models import model_zoo
from data import get_data
from torch.optim import SGD as sgd
from torch.optim import Adam as adam
from checkpoint.find_max_acc import find_max

# ==================================================================
# Parser Initialization
# ==================================================================
print('***** Prepare Data ******')
parser = argparse.ArgumentParser(description='set parama')
parser.add_argument('--cfg', type=str, default=None)
parser.add_argument('--lr', default=0.01, type=float, help='lr')
parser.add_argument('--train_batchsize', default=128, type=int, help='train_batchsize')
parser.add_argument('--test_batchsize', default=128, type=int, help='test_batchsize')
parser.add_argument('--loadModel', default='true', type=str, help='load model parameters')
args = parser.parse_args()
cfg = load_cfg(args.cfg)
args = merge_args_cfg(args, cfg)

if args.k_fold > -1:
    checkpoint_file = args.model_name + '_k{}'.format(args.k_fold)
else:
    checkpoint_file = args.model_name + '_{}'.format(args.data_name)

pathModelParams = './checkpoint/' + checkpoint_file + '/{}.pt'.format(checkpoint_file)
latest_dict_fname = ".".join(pathModelParams.split(".")[:-1]) + "_lastest.pt"
log_filename = './checkpoint/{}/log.txt'.format(checkpoint_file)
path = './checkpoint/' + checkpoint_file
if os.path.isdir(path):
    pass
else:
    os.mkdir(path)

optim_dict = dict(sgd=sgd, adam=adam)
print('***** Parser init done ******')

# ==================================================================
# Prepare Dataset(training & test)
# ==================================================================
print('***** Prepare Data ******')
train_transforms, test_transforms, train_loader, test_loader = get_data(args.data_name, args.train_batchsize,
                                                                        args.test_batchsize, args.k_fold)

print("Dataset done")

# ==================================================================
# Prepare Model
# ==================================================================
print('\n***** Prepare Model *****')

# model = get_model(model_name=args.model_name, num_class=args.num_class)
model = model_zoo(args)

# optim = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
optim = get_optim(args.optim_name, model.parameters(), args.lr, args.weight_decay)
criterion = nn.CrossEntropyLoss().cuda()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=args.EPOCH)

if args.loadModel == 'true':
    model.load_state_dict(torch.load(latest_dict_fname))
model = model.cuda()
cudnn.benchmark = True
if os.path.exists(log_filename):
    best_prec1 = find_max(log_filename)
else:
    best_prec1=0

for epoch in range(0, args.EPOCH):
    train(train_loader, model, criterion, optim, scheduler, epoch)
    prec1, prec5 = validate(test_loader, model, criterion, args.num_class)
    if prec1 > best_prec1:
        torch.save(model.state_dict(), pathModelParams)
        print('Checkpoint saved to {}'.format(pathModelParams))
    torch.save(model.state_dict(), latest_dict_fname)
    print('Save the lastest model to {a},bast prec1:{b}'.format(a=pathModelParams, b=best_prec1))
    with open(log_filename, mode='a', encoding="utf-8") as f:
        f.write(
            f"Epoch[{epoch}/{args.EPOCH}]: Prec@1 {prec1:.3f}; Prec@5 {prec5:.3f}\n"
        )
    best_prec1 = max(prec1, best_prec1)
