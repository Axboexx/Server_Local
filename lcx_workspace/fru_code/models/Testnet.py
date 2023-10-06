"""
@Project ：Server_Local 
@File    ：Testnet.py
@IDE     ：PyCharm 
@Author  ：chengxuLiu
@Date    ：2023/10/1 15:41 
"""
import math
import torchvision
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
import torch.nn as nn
import torchinfo
from torch import Tensor
from typing import List


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class Partial_block(nn.Module):
    def __init__(self, dim):
        super(Partial_block, self).__init__()
        self.div_dim = int(dim / 3)
        self.remainder_dim = dim % 3
        self.p1 = Partial_conv3(self.div_dim, 2)
        self.p2 = Partial_conv3(self.div_dim, 2)
        self.p3 = Partial_conv3(self.div_dim + self.remainder_dim, 2)

    def forward(self, x):
        x1, x2, x3 = torch.split(x, [self.div_dim, self.div_dim, self.div_dim + self.remainder_dim], dim=1)
        print(x1.shape)
        x1 = self.p1(x1)
        x2 = self.p2(x2)
        x3 = self.p3(x3)
        x = torch.cat((x1, x2, x3), 1)
        return x


# FasterNet_Block
class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]
        # conv+BN+ReLu+Conv
        self.mlp = nn.Sequential(*mlp_layer)
        # PConv
        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class BasicStage(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 norm_layer,
                 act_layer,
                 pconv_fw_type
                 ):
        super().__init__()

        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class TestNet(nn.Module):
    def __init__(self, num_classes=92):
        super(TestNet, self).__init__()

        self.Stage1 = nn.Sequential(
            GhostModule(3, 32, kernel_size=1, relu=True),
            GhostModule(32, 64, kernel_size=1, stride=2, relu=True),
            GhostModule(64, 120, relu=True)
        )
        self.Stage2 = nn.Sequential(
            Partial_block(120)
        )
        self.Stage3 = nn.Sequential(
            GhostModule(120, 240, kernel_size=3, stride=2, relu=True),
            Partial_block(240),
            Partial_block(240)
        )
        self.Stage4 = nn.Sequential(
            GhostModule(240, 480, kernel_size=3, relu=True),
            Partial_block(480),
            Partial_block(480),
            Partial_block(480),
            Partial_block(480),
            Partial_block(480),
            Partial_block(480),
            Partial_block(480),
            Partial_block(480)
        )
        self.Stage5 = nn.Sequential(
            GhostModule(480, 960, kernel_size=3, stride=2, relu=True),
            Partial_block(960),
            Partial_block(960)
        )
        self.pool = nn.AvgPool2d(28)
        self.classifier = nn.Sequential(
            nn.Linear(960, 1280, bias=False),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.Stage1(x)
        x = self.Stage2(x)
        x = self.Stage3(x)
        x = self.Stage4(x)
        x = self.Stage5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = TestNet()
    model.eval()
    torchinfo.summary(model, input_size=(3, 3, 224, 224))
