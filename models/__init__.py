"""
@Time : 2023/7/12 17:51
@Author : Axboexx
@File : __init__.py.py
@Software: PyCharm
"""
import timm
from timm import create_model
from torchvision.models import shufflenet_v2_x1_0
from timm.models import create_model
from .mobilevit import *
from .mobilenet_v3 import *
from .mixnet import *
from .Testnet import *
from .shufflenetv1 import *


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
    elif model_name == 'mobilenetv2_140':
        model = create_model(model_name=model_name)
    elif 'TestNet' in model_name:
        model = Testnet_mvit_4M_6G()
    return model


MODEL_ZOO = dict(Testnet_mvit_4M_6G=Testnet_mvit_4M_6G,
                 Testnet_mvit_3M_2G=Testnet_mvit_3M_2G,
                 shufflenetv1=ShuffleNet,
                 shufflenetv2_1_0=shufflenet_v2_x1_0)


def model_zoo(cfg):
    return MODEL_ZOO[cfg.model_name](num_classes=cfg.num_class)
