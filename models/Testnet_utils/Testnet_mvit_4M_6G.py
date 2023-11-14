"""
@Project ：Server_Local 
@File    ：Testnet_mvit_4M_6G.py
@IDE     ：PyCharm 
@Author  ：chengxuLiu
@Date    ：2023/11/7 21:37 
"""
import time

import torch.nn as nn
from .Ghostmodel import *
from .PartialConv import *
from .Transformer import *
from .mobilevit import *


class Testnet_mvit_4M_6G(nn.Module):
    def __init__(self, num_classes=92):
        super(Testnet_mvit_4M_6G, self).__init__()

        # self.pool = 'cls'
        self.to_latent = nn.Identity()

        self.Stage1 = nn.Sequential(
            GhostModule(3, 32, kernel_size=1, stride=2, relu=True),
            Partial_block(32),
            Partial_block(32),

        )
        self.T1 = MobileViTBlock(32, 2, 32, 3, (2, 2), 32, 0.3)

        self.Stage2 = nn.Sequential(
            GhostModule(32, 64, kernel_size=1, stride=2, relu=True),
            Partial_block(64),
            Partial_block(64),
            Partial_block(64),
            Partial_block(64)

        )
        self.T2 = MobileViTBlock(64, 2, 64, 3, (2, 2), 64, 0.3)

        self.Stage3 = nn.Sequential(
            GhostModule(64, 96, kernel_size=3, stride=2, relu=True),
            Partial_block(96),
            Partial_block(96),
            Partial_block(96),
            Partial_block(96),
            Partial_block(96),
            Partial_block(96)
        )
        self.T3 = MobileViTBlock(96, 2, 96, 3, (2, 2), 96, 0.3)

        self.Stage4 = nn.Sequential(
            GhostModule(96, 128, kernel_size=3, relu=True),
            Partial_block(128),
            Partial_block(128),
            Partial_block(128),
            Partial_block(128),
            Partial_block(128),
            Partial_block(128)
        )
        self.T4 = MobileViTBlock(128, 2, 128, 3, (2, 2), 128, 0.3)

        self.Stage5 = nn.Sequential(
            GhostModule(128, 256, kernel_size=3, stride=2, relu=True),
            Partial_block(256),
            Partial_block(256),
            Partial_block(256),
            Partial_block(256)
        )

        self.T5 = MobileViTBlock(256, 2, 256, 3, (2, 2), 256, 0.3)
        self.conv2 = conv_1x1_bn(256, 960)

        self.pool = nn.AvgPool2d(14, 1)
        self.fc = nn.Linear(960, num_classes, bias=False)
        # self.classifier = nn.Sequential(
        #     nn.Linear(960, 1280, bias=False),
        #     nn.BatchNorm1d(1280),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.2),
        #     nn.Linear(1280, num_classes)
        # )

    def forward(self, x):
        x = self.Stage1(x)
        x = self.T1(x)
        x = self.Stage2(x)
        x = self.T2(x)
        x = self.Stage3(x)
        x = self.T3(x)
        x = self.Stage4(x)
        x = self.T4(x)
        x = self.Stage5(x)
        x = self.T5(x)
        x = self.conv2(x)
        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        # x = self.classifier(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary

    model = Testnet_mvit_4M_6G()
    summary(model, input_size=(3, 3, 224, 224))

    # channe1 = 3
    # channe2 = 32
    # resolution = 224
    # PC1 = Partial_conv3(channe1, 3)
    # GM = GhostModule(channe1, channe2, kernel_size=1, relu=True)
    #
    # # summary(PC1, input_size=(1, channe1, resolution, resolution))
    # summary(GM, input_size=(1, channe1, resolution, resolution))
    # x = torch.randn(1, 120, 224, 224)
    # y = x
    # x1, x2, x3 = torch.split(x, [40, 40, 40], dim=1)
    # print(x.shape)
