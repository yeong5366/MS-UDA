
import torch
from torch.nn import  init
import os


def init_weights(net, init_type = 'xavier', init_gain = 1.0, net_type='decoder'):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                if net_type == 'discriminator':
                    init.normal_(m.weight.data,mean=0.0, std=init_gain)
                    #following patchGAN setting
                else:
                    init.normal_(m.weight.data, std=0.001)
            elif init_type == 'xavier':
                if net_type == 'discriminator':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif net_type in ['encoder_th', 'encoder_rgb', 'decoder', 'recon']:
                    init.xavier_uniform_(m.weight.data)
                    #following RTFNet initialization
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find \
                ('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            assert net_type in ['encoder_th', 'discriminator', 'encoder_rgb', 'decoder', 'recon'], 'check net_type'
            if net_type in ['encoder_th', 'encoder_rgb', 'decoder', 'recon']:
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
                #following RTFNet initialization
            elif net_type == 'discriminator':
                # follow patchGAN initialization
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)


    print('initialize %s network with %s' %(net_type, init_type))
    net.apply(init_func)  # apply the initialization function <init_func>

    return net


def init_pretrained_weights(net, pretrained, net_type, init_type='normal'):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')!= -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, std=0.001)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find \
                ('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    print('initialize %s network with %s' % (net_type, init_type))
    net.apply(init_func)  # apply the initialization function <init_func>

    #pretrained_dict.keys(): ['epoch', 'arch', 'num_classes', 'state_dict', 'optimizer', 'mean_iu', 'command', '__metric']
    if os.path.isfile(pretrained):
        pretrained_dict = torch.load(pretrained,
                                     map_location={'cuda:0': 'cpu'})
        pretrained_dict = pretrained_dict['state_dict']
        model_dict = net.state_dict()

        if net_type.find('encoder') !=(-1):
            pretrained_dict={k.replace('backbone.',''):v for k,v in pretrained_dict.items()}
        else:
            pretrained_dict={k.replace('module.',''):v for k,v in pretrained_dict.items()}

        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}
        if pretrained_dict.keys() == model_dict.keys():
            print('all keys of %s are matched with %s' %(net_type, pretrained))

        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
    elif pretrained:
        raise RuntimeError('No such file {}'.format(pretrained))

    return net

def init_net(net, init_type='normal', net_type = 'encoder_rgb', gpu = None, init_gain=1.0, pretrained=True):

    assert net_type in ['encoder_rgb', 'encoder_th', 'discriminator', 'decoder'], 'net_type should be encoder_* | discriminator | decoder'

    if net_type in ['discriminator', 'encoder_th']:
        if len(gpu)>1:
            assert (torch.cuda.is_available())
            net.to(0)  # buffer
            new_gpu = [i for i in range(len(gpu))]
            net = torch.nn.DataParallel(net, new_gpu) # multi-GPUs
            net = init_weights(net, init_type, init_gain, net_type)
        # discriminator and encoder_th do not use pretrained weight
        else:
            assert(torch.cuda.is_available())
            net.to(0)
            net = init_weights(net, init_type, init_gain, net_type)
    elif net_type in ['encoder_rgb', 'decoder']:
        # decoder
        if len(gpu)>1:
            assert(torch.cuda.is_available())

            net.to(0)  # buffer
            new_gpu = [i for i in range(len(gpu))]
            net = torch.nn.DataParallel(net, new_gpu)  # multi-GPUs
            net = init_pretrained_weights(net, pretrained=pretrained, net_type=net_type)
        else:
            assert(torch.cuda.is_available())
            net.to(0)
            net = init_pretrained_weights(net, pretrained=pretrained, net_type=net_type)
    print('initialze %s and wrap it with dataparallel' %net_type)
    return net

