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

SENet in PyTorch.

SENet is the winner of ImageNet-2017. The paper is not released yet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ptflops import get_model_complexity_info

class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        print('~~~~~~~~~~~NEW BLOCK~~~~~~~~~~~~~~')
        out = F.relu(self.bn1(x))
        if hasattr(self,'shortcut'):
            shortcut=self.shortcut(out)
            print(shortcut.shape, 'habsjfn')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        else:
            shortcut=x
            print(shortcut.shape, 'SHORTCUT REG SHAPE')
        out = self.conv1(out)
        print(out.shape, 'POST CONV1')
        out = self.conv2(F.relu(self.bn2(out)))
        print(out.shape, 'POST CONV2')

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        print(w.shape, 'W POST AVG')
        w = F.relu(self.fc1(w))
        print(w.shape, 'W POST FC1')
        w = F.sigmoid(self.fc2(w))
        print(w.shape, 'W POST FC2')
        # Excitation
        out = out * w
        print(out.shape, 'POST OUT*W MULT')
        out += shortcut
        print(out.shape, 'POST SHORTCUT')
        print('~~~~~~~~~~~END BLOCK~~~~~~~~~~~~~~')
        return out


class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SENet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        print(out.shape, 'POST TRUE CONV1')
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        print(out.shape, 'POST FINAL AVG POOL')
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def SENet18(num_classes: int = 10):
    return SENet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)


def test():
    net = SENet18()
    x = torch.randn(1, 3, 32, 32)

    macs, params = get_model_complexity_info(net, (3,32,32), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


    y = net(x)
    print(y.size())


#test()
