"""
@Project ：Server_Local 
@File    ：Testnet_mvit_3M_2G.py
@IDE     ：PyCharm 
@Author  ：chengxuLiu
@Date    ：2023/11/2 21:15 
"""
import torch.nn as nn
from .Ghostmodel import *
from .PartialConv import *
from .Transformer import *
from .mobilevit import *


class Testnet_mvit_3M_2G(nn.Module):
    def __init__(self, num_classes=92):
        super(Testnet_mvit_3M_2G, self).__init__()

        # self.pool = 'cls'
        self.to_latent = nn.Identity()
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(960),
        #     nn.Linear(960, 92)
        # )
        self.Stage1 = nn.Sequential(
            GhostModule(3, 32, kernel_size=1, stride=2, relu=True),
            Partial_block(32),
            Partial_block(32),
            # GhostModule(64, 120, stride=2, relu=True)
        )
        self.Stage2 = nn.Sequential(
            GhostModule(32, 64, kernel_size=1, stride=2, relu=True),
            Partial_block(64),
            Partial_block(64),
            Partial_block(64),
            Partial_block(64)
            # Partial_block(120)
        )
        self.Stage3 = nn.Sequential(
            GhostModule(64, 96, kernel_size=3, stride=2, relu=True),
            Partial_block(96),
            Partial_block(96),
            Partial_block(96),
            Partial_block(96),
            Partial_block(96),
            Partial_block(96)
        )
        self.Stage4 = nn.Sequential(
            GhostModule(96, 128, kernel_size=3, relu=True),
            Partial_block(128),
            Partial_block(128),
            Partial_block(128),
            Partial_block(128),
            Partial_block(128),
            Partial_block(128)
        )
        self.Stage5 = nn.Sequential(
            GhostModule(128, 256, kernel_size=3, stride=2, relu=True),
            Partial_block(256),
            Partial_block(256),
            Partial_block(256),
            Partial_block(256),
            Partial_block(256)
        )

        self.Transformer = MobileViTBlock(256, 2, 256, 3, (2, 2), 256, 0.2)
        self.conv2 = conv_1x1_bn(256, 960)

        self.pool = nn.AvgPool2d(14, 1)
        self.fc = nn.Linear(960, num_classes, bias=False)
        # self.PatchEmbed = PatchEmbedding(embed_size=960, patch_size=3, channels=960, img_size=14)
        # self.Transformer = Transformer(dim=960, depth=1, n_heads=4, mlp_expansions=1, dropout=0.2)

        self.classifier = nn.Sequential(
            nn.Linear(960, 1280, bias=False),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
        # self.classifier=nn.Linear()

    def forward(self, x):
        x = self.Stage1(x)
        x = self.Stage2(x)
        x = self.Stage3(x)
        x = self.Stage4(x)
        x = self.Stage5(x)
        x = self.Transformer(x)
        x = self.conv2(x)
        # x.shape(3,960,14,14)
        x = self.pool(x).view(x.size(0), -1)
        # x = self.fc(x)
        # x = self.PatchEmbed(x)
        # x = self.Transformer(x)
        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        # x = self.to_latent(x)
        # x = self.mlp_head(x)
        # x = self.pool(x)
        # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary

    model = Testnet_mvit_3M_2G()
    summary(model, input_size=(1, 3, 224, 224))

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
