import torch.nn as nn
import functools

from utils.block_module import TransBottleneck


class Decoder(nn.Module):

    def __init__(self, num_classes, inplanes, norm_layer=nn.BatchNorm2d):

        super(Decoder, self).__init__()
        self.inplanes = inplanes

        self.deconv1 = self._make_transpose_layer(TransBottleneck, self.inplanes // 2, 2,
                                                  stride=2, norm_layer=norm_layer)  # using // for python 3.6
        self.deconv2 = self._make_transpose_layer(TransBottleneck, self.inplanes // 2, 2,
                                                  stride=2, norm_layer=norm_layer)  # using // for python 3.6
        self.deconv3 = self._make_transpose_layer(TransBottleneck, self.inplanes // 2, 2,
                                                  stride=2, norm_layer=norm_layer)  # using // for python 3.6
        self.deconv4 = self._make_transpose_layer(TransBottleneck, self.inplanes // 2, 2,
                                                  stride=2, norm_layer=norm_layer)  # using // for python 3.6
        self.deconv5 = self._make_transpose_layer(TransBottleneck, num_classes, 2, stride=2)

    def _make_transpose_layer(self, block, planes, blocks, stride=1, norm_layer=nn.BatchNorm2d):
        upsample = None
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer == nn.InstanceNorm2d or nn.GroupNorm
        else:
            use_bias = norm_layer == nn.InstanceNorm2d or nn.GroupNorm

        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes, kernel_size=2, stride=stride, padding=0, bias=use_bias),
                norm_layer(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=use_bias),
                norm_layer(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes, norm_layer=norm_layer))

        layers.append(block(self.inplanes, planes, stride, upsample, norm_layer=norm_layer))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)

        return x

