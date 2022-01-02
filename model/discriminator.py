
import torch.nn as nn
import functools



class fin_Discriminator(nn.Module):

    def __init__(self, num_classes, ndf=64, norm_layer=nn.BatchNorm2d):

        super(fin_Discriminator, self).__init__()
        # 1/64 --> 1/8
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer == nn.InstanceNorm2d or nn.GroupNorm
        else:
            use_bias = norm_layer == nn.InstanceNorm2d or nn.GroupNorm

        layers = []
        self.layer1 = nn.Sequential(nn.Conv2d(num_classes, ndf, 4, 2, 1, bias=use_bias),
                                    norm_layer(ndf),
                                    nn.LeakyReLU(0.2, True))
        for i in range(3):
            layers += [nn.Conv2d(ndf, ndf * 2, 4, 2, 1,bias=use_bias),
                       norm_layer(ndf*2),
                       nn.LeakyReLU(0.2, True)]
            ndf = ndf * 2
        self.layers = nn.Sequential(*layers)
        self.pred = nn.Conv2d(ndf, 1, 4, 2, 1)

    def forward(self,x):

        x = self.layer1(x)
        x = self.layers(x)
        x = self.pred(x)

        return x.squeeze()

