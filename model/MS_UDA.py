
import torch.nn as nn

from .generator import Generator
from .discriminator import fin_Discriminator
from .Decoder import Decoder
from utils.initial_utils import init_net
from utils.utils import find_norm


class MS_UDA(nn.Module):
    ## resnet pretrained version

    def __init__(self, args, num_layers=50):
        super(MS_UDA, self).__init__()

        self.num_layers = num_layers
        self.inplanes = 2048

        norm = find_norm(args.norm)

        #we initialize the network and wrap with dataparallel/single gpu.
        # encoders(rgb) use pretrained weight
        # decoder uses xavier uniform
        # discriminator is not initialized

        self.net_G_rgb = init_net(Generator(sensor='rgb', num_layers=self.num_layers), init_type=False,
                                  net_type='encoder_rgb', gpu=args.gpus,
                                  init_gain=args.init_gain)
        self.net_G_thermal = init_net(Generator(sensor='thermal', num_layers=self.num_layers), init_type=False,
                                      net_type='encoder_th', gpu=args.gpus)
        self.decoder = init_net(Decoder(args.num_classes, self.inplanes, norm_layer=norm), init_type=args.init_type,
                                net_type='decoder', gpu=args.gpus)
        self.fin_D = init_net(fin_Discriminator(args.num_classes, norm_layer=norm), init_type=args.init_type,
                              net_type='discriminator', gpu=args.gpus)

        #Following RTFNet, we use pretrained encoder and initialize decoder with xavier method.
        #Following patchGAN, we initialize discriminator using xavier method with 0.02 as initial gain


