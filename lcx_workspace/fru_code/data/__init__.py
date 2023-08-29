"""
@Time : 2023/7/12 17:44
@Author : Axboexx
@File : __init__.py.py
@Software: PyCharm
"""
from .fru92 import construct_fru92_data


def get_data(data_name, tr, te):
    if data_name == 'fru92':
        return construct_fru92_data(train_batchsize=tr, test_batchsize=te)
