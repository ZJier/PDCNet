# Author: Junjie Zhang
import torch.nn as nn
import torch
from torch.nn import init
import numpy as np


class PDCblock(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(PDCblock, self).__init__()
        self.growth_rate = growth_rate
        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1_d1 = nn.Conv2d(in_planes, growth_rate, kernel_size=3, dilation = 1, padding=1, bias=False)

        self.bn2_1 = nn.BatchNorm2d(in_planes)
        self.conv2_d1 = nn.Conv2d(in_planes, growth_rate, kernel_size=3, dilation = 1, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(growth_rate)
        self.conv2_d2 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, dilation = 2, padding=2, bias=False)

        self.bn3_1 = nn.BatchNorm2d(in_planes)
        self.conv3_d1 = nn.Conv2d(in_planes, growth_rate, kernel_size=3, dilation = 1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(growth_rate)
        self.conv3_d2 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, dilation = 2, padding=2, bias=False)
        self.bn3_3 = nn.BatchNorm2d(growth_rate)
        self.conv3_d4 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, dilation = 4, padding=4, bias=False)

    def forward(self, x):
        output_1 = self.conv1_d1(self.relu(self.bn1(x)))
        y1 = output_1

        output2_1 = self.conv2_d1(self.relu(self.bn2_1(x)))
        output2_2 = self.conv2_d2(self.relu(self.bn2_2(y1)))
        output2 = output2_1 + output2_2
        y2 = output2

        output3_1 = self.conv3_d1(self.relu(self.bn3_1(x)))
        output3_2 = self.conv3_d2(self.relu(self.bn3_2(y1)))
        output3_3 = self.conv3_d4(self.relu(self.bn3_3(y2)))
        output3 = output3_1 + output3_2 + output3_3
        y3 = output3
        output = torch.cat([y3, y2, y1, x], 1)
        return output


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        return out


class MDCNet(nn.Module):
    def __init__(self, in_channels, num_classes, growth_rate=52, reduction=0.5):
        super(MDCNet, self).__init__()
        num_planes = 2*growth_rate
        self.growth_rate = growth_rate
        self.conv1 = nn.Conv2d(in_channels, num_planes, kernel_size=3, padding=1, bias=False)

        self.multiblock_1 = PDCblock(num_planes, growth_rate)
        num_planes += 3*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.multiblock_2 = PDCblock(num_planes, growth_rate)
        num_planes += 3*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.multiblock_5 = PDCblock(num_planes, growth_rate)
        num_planes += 3*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight,mode='fan_out')
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = np.squeeze(x, axis=1)
        out = self.conv1(out)
        out = self.trans1(self.multiblock_1(out))
        out = self.trans2(self.multiblock_2(out))
        out = self.multiblock_5(out)
        out = self.pooling(F.relu(self.bn(out)))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out