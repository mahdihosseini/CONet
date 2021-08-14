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
from torch.autograd import Variable
from torchviz import make_dot, make_dot_from_trace

from graphviz import Digraph
import re
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Variable
import torchvision.models as models

import torch.onnx
from ptflops import get_model_complexity_info

class BasicBlock(nn.Module):


    def __init__(self, in_planes, intermediate_planes, out_planes,stride=1):
        self.in_planes=in_planes
        self.intermediate_planes=intermediate_planes
        self.out_planes=out_planes

        super(BasicBlock,self).__init__()
        '''if in_planes!=intermediate_planes:
            #print('shortcut_needed')
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
            #print('shortcut_made')
            self.shortcut=nn.Sequential(
                nn.Conv2d(
                        in_planes,
                        out_planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False
                ),
                nn.BatchNorm2d(out_planes),
                #nn.ReLU()
            )

    def forward(self,y):
        x = self.conv1(y)
        #print(x.shape,'post conv1 block')
        x = self.bn1(x)
        x = self.relu(x)
        x = self.bn2(self.conv2(x))
        #print(x.shape,'post conv2 block')
        #if self.shortcut!=nn.Sequential():
            #print('shortcut_made')
        #print(self.shortcut)
        #print(x.shape)
        #print(y.shape)
        #print(self.shortcut(y).shape)
        x += self.shortcut(y)
        #print(x.shape,'post conv3 block')
        x = self.relu(x)
        return x

class ResNet(nn.Module):

    def __init__(self, block, image_channels=3,index=None,num_classes=10):
        super(ResNet, self).__init__()

        ################################################################################## AdaS ##################################################################################
        self.shortcut_1_index = 7 #Number on excel corresponding to shortcut 1
        self.shortcut_2_index = 16 #Number on excel corresponding to shortcut 2
        self.shortcut_3_index = 29 #Number on excel corresponding to shortcut 2
        ####################### O% ########################
        self.superblock1_indexes=[64, 64, 64, 64, 64, 64, 64]
        self.superblock2_indexes=[128, 128, 128, 128, 128, 128, 128, 128]
        self.superblock3_indexes=[256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
        self.superblock4_indexes=[512, 512, 512, 512, 512, 512]
        ####################### 20% #######################
        '''self.superblock1_indexes=[58, 58, 58, 58, 58, 60, 58]
        self.superblock2_indexes=[110, 114, 114, 114, 118, 114, 118, 114]
        self.superblock3_indexes=[218, 226, 226, 226, 228, 226, 228, 226, 228, 226, 228, 226]
        self.superblock4_indexes=[432, 424, 424, 424, 418, 424]'''
        ####################### 40% #######################
        '''self.superblock1_indexes=[54, 54, 54, 52, 54, 54, 54]
        self.superblock2_indexes=[92, 98, 102, 98, 106, 98, 108, 98]
        self.superblock3_indexes=[180, 194, 196, 194, 202, 194, 202, 194, 200, 194, 198, 194]
        self.superblock4_indexes=[354, 336, 336, 336, 324, 336]'''
        ####################### 60% #######################
        '''self.superblock1_indexes=[48, 48, 48, 46, 48, 50, 48]
        self.superblock2_indexes=[74, 84, 88, 84, 96, 84, 98, 84]
        self.superblock3_indexes=[142, 164, 168, 164, 174, 164, 174, 164, 172, 164, 170, 164]
        self.superblock4_indexes=[274, 246, 250, 246, 230, 246]'''
        ####################### 80% #######################
        '''self.superblock1_indexes=[44, 44, 44, 42, 44, 46, 44]
        self.superblock2_indexes=[54, 70, 76, 70, 84, 70, 88, 70]
        self.superblock3_indexes=[104, 132, 138, 132, 148, 132, 148, 132, 144, 132, 140, 132]
        self.superblock4_indexes=[196, 158, 162, 158, 136, 158]'''
        ####################### 100% #######################
        '''self.superblock1_indexes=[38, 38, 38, 36, 38, 40, 38]
        self.superblock2_indexes=[36, 54, 62, 54, 74, 54, 76, 54]
        self.superblock3_indexes=[66, 102, 108, 102, 120, 102, 120, 102, 116, 102, 112, 102]
        self.superblock4_indexes=[116, 70, 74, 70, 42, 70]'''

        if index!=None:
            self.superblock1_indexes=index[0]
            self.superblock2_indexes=index[1]
            self.superblock3_indexes=index[2]
            self.superblock4_indexes=index[3]

        self.index=self.superblock1_indexes+self.superblock2_indexes+self.superblock3_indexes+self.superblock4_indexes

        self.num_classes=num_classes
        self.conv1 = nn.Conv2d(image_channels, self.index [0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.index[0])
        self.network=self._create_network(block)
        self.linear=nn.Linear(self.index[len(self.index)-1],num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu=nn.ReLU()

    def _create_network(self,block):
        output_size=56
        layers=[]
        layers.append(block(self.index[0],self.index[1],self.index[2],stride=1))
        for i in range(2,len(self.index)-2,2):
            #print(self.index [i],self.index [i+1],self.index [i+2],'for loop ',i)
            #if (self.index[i]!=self.index[i+2] or self.index[i]!=self.index[i+1]) and output_size>4:
            if (self.index[i]!=self.index[i+2]):
                stride=2
                output_size=int(output_size/2)
            else:
                stride=1
        #    if i==len(self.index)-4:
            #    self.linear=nn.Linear(self.index[len(self.index)-2],self.num_classes)
            layers.append(block(self.index[i],self.index[i+1],self.index[i+2],stride=stride))
        #    #print(i, 'i')
        #print(len(self.index),'len index')
        return nn.Sequential(*layers)

    def forward(self, y):
        #print(self.index )
        x = self.conv1(y)
        #print(x.shape, 'conv1')
        x = self.bn1(x)
        #print(x.shape, 'bn1')
        x = self.relu(x)
        #print(x.shape, 'relu')
        #x = self.maxpool(x)
        ##print(x.shape, 'max pool')
        x = self.network(x)
        #print(x.shape, 'post bunch of blocks')
        x = self.avgpool(x)
        #print(x.shape, 'post avgpool')
        x = x.view(x.size(0), -1)
        #print(x.shape, 'post reshaping')
        x = self.linear(x)
        #print(x.shape, 'post fc')
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
    #writer = SummaryWriter('runs/resnet34_1')
    net = ResNet34()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

    macs, params = get_model_complexity_info(net, (3,32,32), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    #print(net)
    g=make_dot(y)
    g.view()
    '''

    #g.view()
    torch.save(net.state_dict(),'temp_resnet.onnx')
    dummy_input = Variable(torch.randn(4, 3, 32, 32))
    torch.onnx.export(net, dummy_input, "model.onnx")
    '''

#test()
