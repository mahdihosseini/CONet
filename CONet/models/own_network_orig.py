"""
MIT License
Copyright (c) 2017 liukuang
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
ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Network(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, new_output_sizes=None):
        super(Network, self).__init__()

        #######################  O% ########################
        #self.index=[64,128,256,512]
        ####################### 20% #######################
        #self.index=[54,106,210,414]
        ####################### 40% #######################
        #self.index=[44,84,164,316]
        ####################### 60% #######################
        #self.index=[32,62,120,220]
        ####################### 80% #######################
        #self.index=[32,40,74,122]
        ####################### 100% #######################
        #self.index=[32,18,28,24]

        ####################### OUR OWN #######################
        self.index=[64,64,64,64,64]

        if new_output_sizes!=None:
            self.index=new_output_sizes

        self.in_planes = self.index[0]
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.layer1 = self._make_layer(block, self.index[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.index[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.index[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, self.index[3], num_blocks[3], stride=2)
        self.layer5 = self._make_layer(block, self.index[4], num_blocks[4], stride=2)
        self.linear = nn.Linear(self.index[4], num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def AdaptiveNet(num_classes: int = 10, new_output_sizes=None):
    if new_output_sizes==None:
        return Network(BasicBlock, [3, 3, 3, 3, 3], num_classes=num_classes)
    else:
        return Network(BasicBlock, [3, 3, 3, 3, 3], num_classes=num_classes,new_output_sizes=new_output_sizes)


def test():
    net = AdaptiveNet()
    ##print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    #print(y.shape)
    for param_tensor in net.state_dict():
        if param_tensor.find('conv')==-1:
            continue
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())


#test()
