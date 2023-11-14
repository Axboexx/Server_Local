"""
@Project ：Server_Local 
@File    ：Testnet.py
@IDE     ：PyCharm 
@Author  ：chengxuLiu
@Date    ：2023/10/1 15:41 
"""
from .Testnet_utils.Testnet_mvit_4M_6G import Testnet_mvit_4M_6G
from .Testnet_utils.Testnet_mvit_3M_2G import Testnet_mvit_3M_2G

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
