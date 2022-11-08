import torch
import torch.nn as nn
import torchvision.models as models

import functools




class TransBottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, upsample=None, norm_layer=nn.BatchNorm2d):
        super(TransBottleneck, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer == nn.InstanceNorm2d or nn.GroupNorm
        else:
            use_bias = norm_layer == nn.InstanceNorm2d or nn.GroupNorm

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=use_bias)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn2 = norm_layer(planes)

        if upsample is not None and stride != 1:
            self.conv3 = nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=stride, padding=0, bias=norm_layer)
        else:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=norm_layer)

        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d) \
                    or isinstance(m, nn.InstanceNorm2d) \
                    or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out