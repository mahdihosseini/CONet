'''networks that are light, ~1 million
run baseline and note accuracies.
stepLR w/same learning_rate starting rate = 0.1
FLOPs and MACs, number of learnable parameters, time spent for training,
Number of learnable parameters.
'''

"""
MIT License

Copyright (c) 2020 Mahdi S. Hosseini

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

MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchviz import make_dot, make_dot_from_trace
from ptflops import get_model_complexity_info

import torch
import torch.nn as nn
from torch.nn.modules.module import _addindent
import numpy as np

class Block(nn.Module):
    #expand+depthwise+pointwise

    def __init__(self,in_planes, out_planes_1, out_planes_2, out_planes, stride, shortcut=False):
        self.in_planes=in_planes
        self.out_planes=out_planes
        self.shortcut=shortcut
        super(Block,self).__init__()
        self.stride=stride

        self.conv1=nn.Conv2d(in_planes, out_planes_1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1=nn.BatchNorm2d(out_planes_1)
        self.conv2=nn.Conv2d(out_planes_1, out_planes_2, kernel_size=3, stride=stride, padding=1, groups=out_planes_2, bias=False)
        self.bn2=nn.BatchNorm2d(out_planes_2)
        self.conv3=nn.Conv2d(out_planes_2, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3=nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride==1 and in_planes!=out_planes and shortcut!=False:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self,y):
        x = F.relu(self.bn1(self.conv1(y)))
        #print(x.shape, 'post conv1')
        x = F.relu(self.bn2(self.conv2(x)))
        #print(x.shape, 'post conv2')
        x = self.bn3(self.conv3(x))
        #print(x.shape, 'post conv3')
        #if self.shortcut!=nn.Sequential():
            #print(x.shape, 'out')
            #print(y.shape, 'in')
            #print(self.shortcut(y).shape, 'shortcut_in')
        x = x + self.shortcut(y) if (self.stride == 1) else x
        return x

32, 14, 14, 8, 8, 10, 10, 16, 16, 16, 16, 16, 16, 16, 20, 22, 22, 20, 22, 22, 22, 22, 22, 38, 32, 32, 38, 40, 40, 38, 38, 38, 38, 46, 46, 54, 28, 50, 50, 28, 52, 52, 28, 64, 64, 64, 50, 50, 64, 80, 80, 64, 90, 90, 22, 18, 14
class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        #self.index=[32, 32, 32, (16,'shortcut'), 16, 96, 96, (24,'shortcut'), 24, 144, 144, 24, 144, 144, 32, 192, 192, 32, 192, 192, 32, 192, 192, 64, 384, 384, 64, 384, 384, 64, 384, 384, 64, 384, 384, (96,'shortcut'), 96, 576, 576, 96, 576, 576, 96, 576, 576, 160, 960, 960, 160, 960, 960, 160, 960, 960, (320,'shortcut'), 320, 1280]
        #self.strides_and_short=[(1,True),(1,True),(1,False),(2,False),(1,False),(1,False),(2,False),(1,False),(1,False),(1,False),(1,True),(1,False),(1,False),(2,False),(1,False),(1,False),(1,True)]
        ########################################StepLR Choices#####################################
        ######################################## 0% #############################################
        #self.index=[32, 32, 32, (16,'shortcut'), 16, 96, 96, (24,'shortcut'), 24, 144, 144, 24, 144, 144, 32, 192, 192, 32, 192, 192, 32, 192, 192, 64, 384, 384, 64, 384, 384, 64, 384, 384, 64, 384, 384, (96,'shortcut'), 96, 576, 576, 96, 576, 576, 96, 576, 576, 160, 960, 960, 160, 960, 960, 160, 960, 960, (320,'shortcut'), 320, 1280]
        #self.strides_and_short=[(1,True),(1,True),(1,False),(2,False),(1,False),(1,False),(2,False),(1,False),(1,False),(1,False),(1,True),(1,False),(1,False),(2,False),(1,False),(1,False),(1,True)]
        ######################################## 20% ############################################
        #self.index=[32, 28, 28, (14,'shortcut'), 14, 78, 78, (22,'shortcut'), 20, 118, 118, 20, 118, 118, 30, 158, 158, 30, 158, 158, 30, 158, 158, 58, 314, 314, 58, 316, 316, 58, 314, 314, 58, 316, 316, (88,'shortcut'), 82, 470, 470, 82, 472, 472, 82, 474, 474, 140, 778, 778, 140, 784, 784, 140, 786, 786, (260,'shortcut'), 260, 1026]
        #self.strides_and_short=[(1,True),(1,True),(1,False),(2,False),(1,False),(1,False),(2,False),(1,False),(1,False),(1,False),(1,True),(1,False),(1,False),(2,False),(1,False),(1,False),(1,True)]
        ######################################## 40% ############################################
        #self.index=[32, 24, 24, (12,'shortcut'), 14, 62, 62, (20,'shortcut'), 20, 94, 94, 20, 92, 92, 28, 124, 124, 28, 124, 124, 28, 124, 124, 54, 244, 244, 54, 246, 246, 54, 246, 246, 54, 248, 248, (80,'shortcut'), 68, 366, 366, 68, 366, 366, 68, 372, 372, 122, 596, 596, 122, 608, 608, 122, 612, 612, (200,'shortcut'), 200, 774]
        #self.strides_and_short=[(1,True),(1,True),(1,False),(2,False),(1,False),(1,False),(2,False),(1,False),(1,False),(1,False),(1,True),(1,False),(1,False),(2,False),(1,False),(1,False),(1,True)]
        ######################################## 60% ############################################
        #self.index=[32, 20, 20, (10,'shortcut'), 12, 44, 44, (18,'shortcut'), 18, 68, 68, 18, 68, 68, 24, 90, 90, 24, 90, 90, 24, 90, 90, 48, 172, 172, 48, 178, 178, 48, 176, 176, 48, 182, 182, (70,'shortcut'), 56, 260, 260, 56, 262, 262, 56, 268, 268, 102, 414, 414, 102, 432, 432, 102, 438, 438, (142,'shortcut'), 138, 520]
        #self.strides_and_short=[(1,True),(1,True),(1,False),(2,False),(1,False),(1,False),(2,False),(1,False),(1,False),(1,False),(1,True),(1,False),(1,False),(2,False),(1,False),(1,False),(1,True)]
        ######################################## 80% ############################################
        #self.index=[32, 16, 16, (10,'shortcut'), 10, 28, 28, (16,'shortcut'), 16, 42, 42, 16, 42, 42, 22, 56, 56, 22, 56, 56, 22, 56, 56, 42, 102, 102, 42, 110, 110, 42, 106, 106, 42, 114, 114, (62,'shortcut'), 42, 156, 156, 42, 158, 158, 42, 166, 166, 84, 232, 232, 84, 256, 256, 84, 264, 264, (82,'shortcut'), 78, 268]
        #self.strides_and_short=[(1,True),(1,True),(1,False),(2,False),(1,False),(1,False),(2,False),(1,False),(1,False),(1,False),(1,True),(1,False),(1,False),(2,False),(1,False),(1,False),(1,True)]
        ######################################## 100% ###########################################
        self.index=[32, 14, 14, (8,'shortcut'), 8, 10, 10, (16,'shortcut'), 16, 16, 16, 16, 16, 16, 20, 22, 22, 20, 22, 22, 20, 22, 22, 38, 32, 32, 38, 40, 40, 38, 38, 38, 38, 46, 46, (54,'shortcut'), 28, 50, 50, 28, 52, 52, 28, 64, 64, 64, 50, 50, 64, 80, 80, 64, 90, 90, (22,'shortcut'), 18, 14]
        self.strides_and_short=[(1,True),(1,True),(1,False),(2,False),(1,False),(1,False),(2,False),(1,False),(1,False),(1,False),(1,True),(1,False),(1,False),(2,False),(1,False),(1,False),(1,True)]
        #########################################################################################

        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, self.index[0], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.index[0])

        #self.layers = self._make_layers(in_planes=32)
        self.layers=self._create_network(Block)

        self.conv2 = nn.Conv2d(self.index[len(self.index)-2], self.index[len(self.index)-1], kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(self.index[len(self.index)-1])
        self.linear = nn.Linear(self.index[len(self.index)-1], num_classes)

    def _create_network(self,block):
        layers=[]
        input_res=32
        i=0
        stride_i=0
        while i<len(self.index)-4:
            if isinstance(self.index[i],int)!=True:
                i+=1
            else:
                if isinstance(self.index[i+3],tuple)==True:
                    layers.append(block(self.index[i],self.index[i+1],self.index[i+2],self.index[i+4],1,shortcut=True))
                    i+=4
                    stride_i+=1
                else:
                    input_res=input_res/2
                    stride = self.strides_and_short[stride_i][0]#if input_res>2 else 1
                    #print(stride, stride_i, 'stride choice')
                    #shortcut=self.strides_and_short[other_i][1]
                    layers.append(block(self.index[i],self.index[i+1],self.index[i+2],self.index[i+3],stride))
                    i+=3
                    stride_i+=1
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
'''
    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)
'''

def test():
    #Baseline: 2296922, 20%: 1613814, 40%:
    net = MobileNetV2()

    macs, params = get_model_complexity_info(net, (3,32,32), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(pytorch_total_params)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

#test()
