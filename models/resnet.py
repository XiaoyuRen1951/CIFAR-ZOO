# -*-coding:utf-8-*-

import torch.nn as nn

__all__ = ["resnet20", "resnet20_bnk", "resnet29_bnk", "resnet32", "resnet44", "resnet47_bnk", "resnet56", "resnet56_bnk", "resnet68", "resnet74_bnk", "resnet80", "resnet92", "resnet92_bnk", "resnet110", "resnet110_bnk", "resnet1202"]


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv_1 = conv3x3(inplanes, planes, stride)
        self.bn_1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = conv3x3(planes, planes)
        self.bn_2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)

        out = self.conv_2(out)
        out = self.bn_2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv_1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(planes)
        self.conv_2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn_2 = nn.BatchNorm2d(planes)
        self.conv_3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)

        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.relu(out)

        out = self.conv_3(out)
        out = self.bn_3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, depth, layers, num_classes, block_name="BasicBlock"):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name == "BasicBlock":
            assert (
                depth - 2
            ) % 6 == 0, "depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202"
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name == "Bottleneck":
            assert (
                depth - 2
            ) % 9 == 0, "depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199"
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError("block_name shoule be Basicblock or Bottleneck")

        self.inplanes = 16
        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.stage_1 = self._make_layer(block, 16, layers[0])
        self.stage_2 = self._make_layer(block, 32, layers[1], stride=2)
        self.stage_3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_normal(m.weight.data)
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)  # 32x32

        x = self.stage_1(x)  # 32x32
        x = self.stage_2(x)  # 16x16
        x = self.stage_3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet20(num_classes):
    return ResNet(depth=20, layers=[3,3,3], num_classes=num_classes)

def resnet20_bnk(num_classes):
        return ResNet(depth=20, layers=[2,2,2], num_classes=num_classes, block_name="Bottleneck")

def resnet29_bnk(num_classes):
    return ResNet(depth=29, layers=[3,3,3], num_classes=num_classes, block_name="Bottleneck")

def resnet32(num_classes):
    return ResNet(depth=32, layers=[5,5,5], num_classes=num_classes)

def resnet44(num_classes):
    return ResNet(depth=44, layers=[7,7,7], num_classes=num_classes)

def resnet47_bnk(num_classes):
    return ResNet(depth=47, layers=[5,5,5], num_classes=num_classes, block_name="Bottleneck")

def resnet56(num_classes):
    return ResNet(depth=56, layers=[9,9,9], num_classes=num_classes)

def resnet56_bnk(num_classes):
    return ResNet(depth=56, layers=[6,6,6], num_classes=num_classes, block_name="Bottleneck")

def resnet68(num_classes):
    return ResNet(depth=68, layers=[11,11,11], num_classes=num_classes)

def resnet74_bnk(num_classes):
    return ResNet(depth=74, layers=[8,8,8], num_classes=num_classes, block_name="Bottleneck")

def resnet80(num_classes):
    return ResNet(depth=80, layers=[13,13,13], num_classes=num_classes)

def resnet92(num_classes):
    return ResNet(depth=92, layers=[15,15,15], num_classes=num_classes)

def resnet92_bnk(num_classes):
    return ResNet(depth=110, layers=[10,10,10], num_classes=num_classes, block_name="Bottleneck")

def resnet110(num_classes):
    return ResNet(depth=110, layers=[18,18,18], num_classes=num_classes)

def resnet110_bnk(num_classes):
    return ResNet(depth=110, layers=[12,12,12], num_classes=num_classes, block_name="Bottleneck")

def resnet1202(num_classes):
    return ResNet(depth=1202, layers=[200,200,200], num_classes=num_classes)
