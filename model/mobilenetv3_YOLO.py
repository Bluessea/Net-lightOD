'''MobileNetV3 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn import init



class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class SkipBlock(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1):
        super(SkipBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((112,112))

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size,stride=2,padding=kernel_size//2,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size,stride=2,padding=kernel_size//2,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size, stride=2, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
        )


    def forward(self, x):
        out = self.avgpool(x)
        out1 = self.conv1(out)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        return out3






"""
class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out
"""


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear

        self.shortcutMid = nn.Sequential()
        if in_size != expand_size:
            self.shortcutMid = nn.Sequential(
                nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(expand_size),
            )

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):

        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out))) + self.shortcutMid(x) if self.stride==1 else self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNetV3_Scale3_large(nn.Module):
    def __init__(self):
        super(MobileNetV3_Scale3_large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.scale_1 = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
        )
        self.scale_2 = nn.Sequential(
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
        )

        self.scale_3 = nn.Sequential(
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )
    def forward(self, x):
        x = self.hs1(self.bn1(self.conv1(x)))
        out3 = self.scale_1(x)
        out2 = self.scale_2(out3)
        out1 = self.scale_3(out2)
        return out1, out2, out3


def YOLOFPN(x):
    if isinstance(x, tuple):
        x0, x_skip = x
        x = Block(3, list(x0.size())[1], 288, list(x0.size())[1], hswish(), None, 1)(x0)
        x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
        # 维度拼接
        x = torch.cat([x,x_skip],dim=1)
        # x = tf.keras.layers.Concatenate()([x, x_skip])

    x = Block(3, list(x.size())[1], 288, 96, hswish(), None, 1)(x)
    x = Block(3, list(x.size())[1], 288, 96, hswish(), None, 1)(x)
    return x

def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

def YOLOHead(in_chnneals,anc_per, num_classes):
    # m = nn.Sequential(
    #     conv2d(in_filters, filters_list[0], 3),
    #     nn.Conv2d(filters_list[0], filters_list[1], 1),
    # )
    x = nn.Sequential(
        ConvBlock(in_chnneals, out_channels=288, kernel_size=3, stride=1, bias=False, bn=True, act="hswish"),
        ConvBlock(288, out_channels=anc_per * (num_classes + 5), kernel_size=1, stride=1, bias=True, bn=False,act=None)
    )
    # x = ConvBlock(list(x.size())[1],out_channels=288,kernel_size=3,stride=1,bias=False,bn=True,act="hswish")(x)
    # x = ConvBlock(list(x.size())[1],out_channels=anc_per * (num_classes + 5),kernel_size=1,stride=1,bias=True,bn=False,act=None)(x)
    return x

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,bias,bn,act):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,bias=bias,padding=(kernel_size-1)//2 if kernel_size else 0)
        self.norm = nn.BatchNorm2d(out_channels) if bn else Identity()
        _act = {
            "hswish": hswish(),
            "hsigmoid": hsigmoid()
        }
        self.activate = _act[act] if act else Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activate(x)
        return x


class MobileYOLONet(nn.Module):
    def __init__(self,training=True):
        super(MobileYOLONet, self).__init__()

        self.training = training
        self.backbone = MobileNetV3_Scale3_large()

        num_anchors = 3
        num_classes = 17
        self.head_1 = YOLOHead(96,num_anchors,num_classes)  # 80 160

        self.head_2 = YOLOHead(96, num_anchors, num_classes)  # 80 16

        self.head_3 = YOLOHead(96, num_anchors, num_classes)  # 80 16

    def forward(self, x):
        s1, s2, s3 = self.backbone(x)

        x = YOLOFPN(s1)
        out1 = self.head_1(x)

        x = YOLOFPN((x,s2))
        out2 = self.head_2(x)

        x = YOLOFPN((x,s3))
        out3 = self.head_3(x)

        if self.training:
            return out1, out2, out3

    # x = YoloFPN(s1)
    # num_anchors = 3
    # num_classes = 17
    # output_1 = YOLOHead(x,num_anchors,num_classes)
    #
    # x = YoloFPN(s2)
    # num_anchors = 3
    # num_classes = 17
    # output_2 = YOLOHead(x, num_anchors, num_classes)
    #
    # x = YoloFPN(s3)
    # num_anchors = 3
    # num_classes = 17
    # output_3 = YOLOHead(x, num_anchors, num_classes)

class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            # Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            # Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )
        self.skipblock = SkipBlock(16,160)

        self.bneck1 = nn.Sequential(
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )



        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out1 = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out1)

        # 加入输入和最后一层的冗余特征框
        skipbolck = self.skipblock(out1)
        out = self.bneck1(out + skipbolck)

        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out



class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            #  kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )


        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(576, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out



def test():
    net = MobileNetV3_Small()
    x = torch.randn(2,3,224,224)
    y = net(x)
    print(y.size())

# test()
if __name__ == '__main__':
    test()