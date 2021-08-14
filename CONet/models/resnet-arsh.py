"""
MIT License

Copyright (c) 2017 liukuang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software withx restriction, including withx limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHx WARRANTY OF ANY KIND, Ex PRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
x OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, x iangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arx iv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

convCount=0

class BasicBlock(nn.Module):

    def __init__(self,in_planes,intermediate_planes, out_planes,stride=1):
        self.in_planes=in_planes
        self.intermediate_planes=intermediate_planes
        self.out_planes=out_planes

        super(BasicBlock,self).__init__()
        '''if in_planes!=intermediate_planes:
            print('shortcut_needed')
            stride=2
        else:
            stride=stride'''
        self.conv1=nn.Conv2d(
                in_planes,
                intermediate_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
        )
        self.bn1=nn.BatchNorm2d(intermediate_planes)
        self.conv2=nn.Conv2d(
                intermediate_planes,
                out_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
        )
        self.bn2=nn.BatchNorm2d(out_planes)
        self.relu=nn.ReLU()
        self.shortcut=nn.Sequential()
        if stride!=1 or in_planes!=out_planes:
            print('shortcut_made')
            self.shortcut=nn.Sequential(
                nn.Conv2d(
                        in_planes,
                        out_planes,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        bias=False
                ),
                nn.BatchNorm2d(out_planes),
                nn.ReLU()
            )

    def forward(self,y):
        '''if x.shape[2]<=4:
            self.conv1=nn.Conv2d(
                    self.in_planes,
                    self.intermediate_planes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
            )
            self.shortcut=nn.Sequential(
                nn.Conv2d(
                        self.in_planes,
                        self.out_planes,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False
                ),
                nn.BatchNorm2d(self.out_planes),
                nn.ReLU()
            )'''
        x = self.conv1(y)
        print(x.shape,'post conv1 block')
        x = self.bn1(x)
        x = self.relu(x)
        x = self.bn2(self.conv2(x))
        print(x.shape,'post conv2 block')
        if self.shortcut!=nn.Sequential():
            print('shortcut_made')
        print(self.shortcut)
        x += self.shortcut(y)
        print(x.shape,'post conv3 block')
        x = self.relu(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, image_channels,num_classes=10):
        super(ResNet, self).__init__()
        #Proposed
        self.index=[64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        #Original
        self.temp=[64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512]
        #self.index  = [64, 64, 96, 96, 108, 108, 128, 128, 140, 140, 150, 150, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        #self.index = [32, 32, 34, 34, 34, 34, 34, 64, 66, 56, 66, 64, 66, 62, 64, 62, 122, 128, 110, 126, 120, 126, 118, 124, 118, 122, 118, 124, 114, 234, 232, 220, 226, 210, 216, 210]
        #self.index = [32, 22, 24, 22, 22, 22, 24, 42, 46, 32, 44, 42, 46, 42, 44, 40, 78, 84, 60, 82, 76, 84, 72, 80, 72, 78, 72, 78, 66, 142, 140, 122, 132, 110, 118, 108]

        print(self.index)
        #self.index_temp=self.index
        #self.index_temp=[64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        #self.index =self.index_temp
        '''if len(self.index)%2==0:
            print(self.index)
            length=len(self.index)-1
            self.index+=[self.index[length]]'''
        self.num_classes=num_classes
        #self.in_planes = 64
        self.conv1 = nn.Conv2d(image_channels, self.index [0], kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.index[0])
        #self.block1=self._make_block(block,self.index [0],self.index [1],self.index [2],stride=1)
        self.network=self._create_network(block)
        self.linear=nn.Linear(self.index[len(self.index )-1],num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu=nn.ReLU()

    def _create_network(self,block):
        output_size=56
        layers=[]
        layers.append(block(self.index[0],self.index[1],self.index[2],stride=1))
        for i in range(2,len(self.index)-2,2):
            print(self.index [i],self.index [i+1],self.index [i+2],'for loop ',i)
            if (self.index[i]!=self.index[i+2] or self.index[i]!=self.index[i+1]) and output_size>4:
                stride=2
                output_size=int(output_size/2)
            else:
                stride=1
        #    if i==len(self.index)-4:
            #    self.linear=nn.Linear(self.index[len(self.index)-2],self.num_classes)
            layers.append(block(self.index[i],self.index[i+1],self.index[i+2],stride=stride))
        #    print(i, 'i')
        print(len(self.index),'len index')
        return nn.Sequential(*layers)

    '''
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.ex pansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.ex pansion
        return nn.Sequential(*layers)'''

    def forward(self, x):
        print(self.index )
        x = self.conv1(x)
        print(x.shape, 'conv1')
        x = self.bn1(x)
        print(x.shape, 'bn1')
        x = self.relu(x)
        print(x.shape, 'relu')
        x = self.maxpool(x)
        print(x.shape, 'max pool')
        x = self.network(x)
        print(x.shape, 'post bunch of blocks')
        x = self.avgpool(x)
        print(x.shape, 'post avgpool')
        x = x.view(x.size(0), -1)
        print(x.shape, 'post reshaping')
        x = self.linear(x)
        print(x.shape, 'post fc')
        return x


def ResNet18(num_classes: int = 10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes = 10):
    return ResNet(BasicBlock, 3, num_classes=10)


def ResNet50(num_classes: int = 10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes: int = 10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes: int = 10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


def test():
    net = ResNet34()
    print(net)
    y = net(torch.randn(1, 3, 224, 224))
    print(y.size())

#test()
