"""
@Time : 2023/7/12 17:54
@Author : Axboexx
@File : main.py
@Software: PyCharm
"""
import argparse

import torch.nn as nn
import torch
from torch.backends import cudnn
from main_lib import if_mkdir
from main_lib import train
from main_lib import validate
from models import get_model
from data import get_data

# num_class = 92
# optim_name = 'sgd'
# model_name = 'mobilenetv3_small'
# data_name = 'fru92'
# train_batchsize = 64
# test_batchsize = 64
# lr = 0.01
# momentum = 0.9
# EPOCH = 300


# ==================================================================
# Parser Initialization
# ==================================================================
print('***** Prepare Data ******')
parser = argparse.ArgumentParser(description='set parama')
parser.add_argument('--EPOCH', default=300, type=int, help="EPOCH")
parser.add_argument('--lr', default=0.01, type=float, help="learning rate")
parser.add_argument('--num_class', default=92, type=int, help="the class number of the dataset")
parser.add_argument('--data_name', default='fru92', type=str, help="dataset choice")
parser.add_argument('--optim_name', default='sgd', type=str, help="the class of the optim")
parser.add_argument('--model_name', default='mobilenetv3_small', type=str, help="model choice")
parser.add_argument('--train_batchsize', default=64, type=int, help="training batch size")
parser.add_argument('--test_batchsize', default=64, type=int, help="testing batch size")
parser.add_argument('--loadModel', default='true', type=str, help='load model parameters')
args = parser.parse_args()
pathModelParams = './checkpoint/{a}/{a}.pt'.format(a=args.model_name)
latest_dict_fname = ".".join(pathModelParams.split(".")[:-1]) + "_lastest.pt"
log_filename = './checkpoint/{}/log.txt'.format(args.model_name)
path = './checkpoint/{}'.format(args.model_name)
optim_dict = dict(sgd='sgd', adam='adam')
print('***** Parser init done ******')

# ==================================================================
# Prepare Dataset(training & test)
# ==================================================================
print('***** Prepare Data ******')
train_transforms, test_transforms, train_loader, test_loader = get_data(args.data_name, args.train_batchsize,
                                                                        args.test_batchsize)

print("Dataset done")

# ==================================================================
# Prepare Model
# ==================================================================
print('\n***** Prepare Model *****')

model = get_model(model_name=args.model_name, num_class=args.num_class)
optim = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss().cuda()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=args.EPOCH)

if args.loadModel == 'true':
    model.load_state_dict(torch.load(pathModelParams))
model = model.cuda()
cudnn.benchmark = True

best_prec1 = 0
if_mkdir(path)
for epoch in range(0, args.EPOCH):
    train(train_loader, model, criterion, optim, scheduler, epoch)
    prec1, prec5 = validate(test_loader, model, criterion, args.num_class)
    is_best = prec1 > best_prec1
    if prec1 > best_prec1:
        torch.save(model.state_dict(), pathModelParams)
        print('Checkpoint saved to {}'.format(pathModelParams))
    torch.save(model.state_dict(), latest_dict_fname)
    print('Save the lastest model to {}'.format(pathModelParams))
    with open(log_filename, mode='a', encoding="utf-8") as f:
        f.write(
            f"Epoch[{epoch}/{args.EPOCH}]: Prec@1 {prec1:.3f}; Prec@5 {prec5:.3f}\n"
        )
    best_prec1 = max(prec1, best_prec1)
