"""This is the CIFAR-10 version of PyramidNet from the implementation of https://github.com/dyhan0920/PyramidNet-PyTorch."""
"""
MIT License

Copyright (c) 2019 Dongyoon Han

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import math


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(torch.nn.Module):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.shape[2:4]
        else:
            shortcut = x
            featuremap_size = out.shape[2:4]

        batch_size = out.shape[0]
        residual_channel = out.shape[1]
        shortcut_channel = shortcut.shape[1]

        if residual_channel != shortcut_channel:
            padding = shortcut.new_zeros(batch_size, residual_channel - shortcut_channel, *featuremap_size)
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut

        return out


class Bottleneck(torch.nn.Module):
    outchannel_ratio = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(inplanes)
        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, (planes * 1), kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d((planes * 1))
        self.conv3 = torch.nn.Conv2d((planes * 1), planes * Bottleneck.outchannel_ratio, kernel_size=1, bias=False)
        self.bn4 = torch.nn.BatchNorm2d(planes * Bottleneck.outchannel_ratio)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = self.bn4(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.shape[2:4]
        else:
            shortcut = x
            featuremap_size = out.shape[2:4]

        batch_size = out.shape[0]
        residual_channel = out.shape[1]
        shortcut_channel = shortcut.shape[1]

        if residual_channel != shortcut_channel:
            padding = shortcut.new_zeros(batch_size, residual_channel - shortcut_channel, *featuremap_size)
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut

        return out


class PyramidNet(torch.nn.Module):

    def __init__(self, depth, alpha, num_channels, num_classes, bottleneck=False):
        super().__init__()

        self.inplanes = 16
        if bottleneck:
            n = int((depth - 2) / 9)
            block = Bottleneck
        else:
            n = int((depth - 2) / 6)
            block = BasicBlock

        self.addrate = alpha / (3 * n * 1.0)

        self.input_featuremap_dim = self.inplanes
        self.conv1 = torch.nn.Conv2d(num_channels, self.input_featuremap_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(self.input_featuremap_dim)

        self.featuremap_dim = self.input_featuremap_dim
        self.layer1 = self.pyramidal_make_layer(block, n)
        self.layer2 = self.pyramidal_make_layer(block, n, stride=2)
        self.layer3 = self.pyramidal_make_layer(block, n, stride=2)

        self.final_featuremap_dim = self.input_featuremap_dim
        self.bn_final = torch.nn.BatchNorm2d(self.final_featuremap_dim)
        self.relu_final = torch.nn.ReLU(inplace=True)
        self.avgpool = torch.nn.AvgPool2d(8)
        self.fc = torch.nn.Linear(self.final_featuremap_dim, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def pyramidal_make_layer(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1:  # or self.inplanes != int(round(featuremap_dim_1st)) * block.outchannel_ratio:
            downsample = torch.nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)

        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, downsample))
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(block(int(round(self.featuremap_dim)) *
                                block.outchannel_ratio, int(round(temp_featuremap_dim)), 1))
            self.featuremap_dim = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
