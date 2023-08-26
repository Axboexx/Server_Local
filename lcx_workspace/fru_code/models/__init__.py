"""
@Time : 2023/7/12 17:51
@Author : Axboexx
@File : __init__.py.py
@Software: PyCharm
"""
from .mobilevit import *
from .mobilenet_v3 import *
from .mixnet import *


def get_model(model_name, num_class):
    model = None
    if model_name == 'mobilevit_s':
        model = mobilevit_s(num_class=num_class)
    elif model_name == 'mobilevit_xs':
        model = mobilevit_xs(num_class=num_class)
    elif model_name == 'mobilevit_xxs':
        model = mobilevit_xxs(num_class=num_class)
    elif model_name == 'mobilenetv3_small':
        model = MobileNetV3_Large(num_classes=num_class)
    elif model_name == 'mobilenetv3_large':
        model = MobileNetV3_Small(num_classes=num_class)
    elif model_name == 'mixnet_s':
        model = MixNet(model_name, num_classes=num_class)
    elif model_name == 'mixnet_m':
        model = MixNet(model_name, num_classes=num_class)
    elif model_name == 'mixnet_l':
        model = MixNet(model_name, num_classes=num_class)
    return model
