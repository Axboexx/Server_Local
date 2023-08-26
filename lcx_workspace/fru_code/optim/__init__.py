"""
@Time : 2023/7/12 17:53
@Author : Axboexx
@File : __init__.py.py
@Software: PyCharm
"""
import torch.optim.sgd as sgd
import torch.optim.adam as adam


def get_optim(optime_name, param, lr, momentum, weight_decay=0.00001):
    if optime_name == 'sgd':
        return sgd(param, lr, momentum, weight_decay=weight_decay)
    if optime_name == 'adam':
        return adam(param, lr)
