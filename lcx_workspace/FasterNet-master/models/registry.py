import torch.nn as nn
from timm.models.registry import register_model
from .fasternet import FasterNet
from .alexnet import AlexNet_i


@register_model
def fasternet(**kwargs):
    model = FasterNet(**kwargs)
    return model


@register_model
def alexnet():
    model = AlexNet_i()
    return model
