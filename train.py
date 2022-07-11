
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from tensorboardX import SummaryWriter

import os,sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


from utils.utils import set_requires_grad, adjust_learning_rate, \
    visualize, load_state_from_model

import argparse
import time
import torch
from datetime import datetime
import numpy as np

from options.print import print_options
from model.MS_UDA import MS_UDA
from utils.augmentation import RandomFlip_KP_fake, RandomFlip_KP, RandomFlip_MF, RandomFlip_MF_fake

from loss.loss import loss_calc

from options.parser_MSUDA import _Parser

parser = argparse.ArgumentParser(description='default for MSG models')
parser.add_argument('--dataset', default='KPdataset', help='KPdataset|MFdataset')
parser.add_argument('--is_continue', default="store_true", help='restart the training')
parser.add_argument('--start_epo', default=0)
parser.add_argument('--fake', default="store_true", help='use fake thermal image or not')

args = parser.parse_args()

if args.dataset =='KPdataset':
    if args.fake:
        parser.add_argument('--fake_prob', default=0.5, help='how many fake thermal images in whole training dataset')
        parser.add_argument('--config', '-c', default='configs/KPD_fake.yaml', help='path to the config file')
        augmentation_methods = [RandomFlip_KP_fake(prob=0.5)]
        from datasets.KPD_fake import KP_dataset as dataset
    else:
        parser.add_argument('--config', '-c',default='configs/KPD.yaml',help='path to the config file')
        augmentation_methods = [RandomFlip_KP(prob=0.5)]
        from datasets.KPD import KP_dataset as dataset
    NUM_DATASET = 3283
elif args.dataset =='MFdataset':
    if args.fake:
        parser.add_argument('--fake_prob', default=0.5, help='how many fake thermal images in whole training dataset')
        parser.add_argument('--config', '-c', default='configs/MFD_fake.yaml', help='path to the config file')
        augmentation_methods = [RandomFlip_MF_fake(prob=0.5)]
        from datasets.MFD_fake import MF_dataset as dataset
    else:
        parser.add_argument('--config', '-c', default='configs/MFD.yaml', help='path to the config file')
        augmentation_methods = [RandomFlip_MF(prob=0.5)]
        from datasets.MFD import MF_dataset as dataset
    NUM_DATASET = 820

Parser = _Parser()
args = Parser.gather_option(args, parser)
print_options(args, parser)

# set cuda environment
if len(args.gpus) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)
elif len(args.gpus) == 1:
    single_gpu = args.gpus[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(single_gpu)


if __name__ == '__main__':

    cudnn.enabled = True

    if args.dataset == 'KPdataset':
        trainloader = DataLoader(
            dataset(args.root_dir + args.data_dir, split='day', label_folder='pseudo_KP',transform=augmentation_methods),
            batch_size=args.train_batch,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False)
    else:
        trainloader = DataLoader(
            dataset(args.root_dir + args.data_dir, 'day', True, augmentation_methods),
            batch_size=args.train_batch,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False)

    model_name = args.model
    if model_name == 'MS_UDA':
        model = MS_UDA(args, num_layers=args.num_layers)  # initialize model, send model to device

    log_dir = os.path.join(args.root_dir, model_name + args.log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if os.path.exists(log_dir) and args.tensorboard:
        writer = SummaryWriter(log_dir)
    if os.path.exists(log_dir):
        record_filename = os.path.join(log_dir, 'loss_record_per_iter.txt')
        record_epoch_filename = os.path.join(log_dir, 'loss_recopord_per_epoch.txt')
        with open(record_filename, 'a') as newtxt:
            newtxt.write(str(datetime.today()) + '\n')
        with open(record_epoch_filename, 'a') as epochtxt:
            epochtxt.write(str(datetime.today()) + '\n')

    check_dir = os.path.join(args.root_dir, model_name + args.check_dir)
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)

    if args.is_continue:
        model_file = os.path.join(check_dir, model_name + '_' + str(args.start_epo) + '.pth')
        pretrained_weight = torch.load(model_file, map_location=lambda storage, loc: storage.cuda(0))
        load_state_from_model(pretrained_weight, model, args.gpus)

    G1 = model.net_G_rgb
    G2 = model.net_G_thermal
    fin_D = model.fin_D
    Decoder = model.decoder

    optimizer_G1 = optim.SGD(G1.parameters(), lr=args.base_lr_G, momentum=args.momentum_G,
                             weight_decay=args.weight_decay_G)
    optimizer_G2 = optim.SGD(G2.parameters(), lr=args.base_lr_G, momentum=args.momentum_G,
                             weight_decay=args.weight_decay_G)
    optimizer_fin_D = optim.Adam(fin_D.parameters(), lr=args.base_lr_D, betas=(0.9, 0.99))
    optimizer_Dec = optim.SGD(Decoder.parameters(), lr=args.base_lr_Dec, momentum=args.momentum_Dec,
                              weight_decay=args.weight_decay_Dec)

    if args.is_continue:
        G1_optim = os.path.join(check_dir, model_name + '_' + str(args.start_epo) + '_G1.optim')
        G2_optim = os.path.join(check_dir, model_name + '_' + str(args.start_epo) + '_G2.optim')
        Dec_optim = os.path.join(check_dir, model_name + '_' + str(args.start_epo) + '_Dec.optim')
        fin_D_optim = os.path.join(check_dir, model_name + '_' + str(args.start_epo) + '_fin_D.optim')
        load_G1 = torch.load(G1_optim, map_location=lambda storage, loc: storage.cuda(0))
        optimizer_G1.load_state_dict(load_G1)
        del load_G1
        load_G2 = torch.load(G2_optim, map_location=lambda storage, loc: storage.cuda(0))
        optimizer_G2.load_state_dict(load_G2)
        del load_G2
        load_Dec = torch.load(Dec_optim, map_location=lambda storage, loc: storage.cuda(0))
        optimizer_Dec.load_state_dict(load_Dec)
        del load_Dec
        load_fin_D = torch.load(fin_D_optim, map_location=lambda storage, loc: storage.cuda(0))
        optimizer_fin_D.load_state_dict(load_fin_D)
        del load_fin_D

    # set training model
    G1.train()
    G2.train()
    fin_D.train()
    Decoder.train()

    source_label = 0  # rgb image
    target_label = 1  # thermal image

    source_label_tensor = torch.tensor(source_label, dtype=torch.float)
    target_label_tensor = torch.tensor(target_label, dtype=torch.float)

    total_iters = 0
    max_iters = NUM_DATASET * args.max_epoch

    # loss between pred1<->label: loss_calc(crossentropy2d)
    if args.gan_type == 'Vanilla':
        adv_loss = nn.BCEwithLogits()
    elif args.gan_type == 'LS':
        adv_loss = nn.MSELoss()
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma

    max_iters = NUM_DATASET * args.max_epoch

    if args.is_continue:
        start_epoch = args.start_epo + 1
        total_iters = start_epoch * NUM_DATASET
    else:
        start_epoch = 0
        total_iters = 0

    ask_fake = args.fake
    ask_dataset = True if args.dataset=='KPdataset' else False

    for epoch in range(start_epoch, args.max_epoch):
        epoch_start_time = time.time()
        LOSS_G1_VALUE = 0
        LOSS_G2_VALUE_adv = 0
        LOSS_G2_VALUE_seg = 0
        LOSS_DECODER_VALUE_pred1 = 0
        LOSS_DECODER_VALUE_pseudo = 0
        LOSS_DECODER_VALUE_adv = 0
        LOSS_FIN_D_VALUE = 0

        for i, batch in enumerate(trainloader):
            iter_start_time = time.time()

            optimizer_G1.zero_grad()
            optimizer_G2.zero_grad()
            optimizer_fin_D.zero_grad()
            optimizer_Dec.zero_grad()

            g_lr = adjust_learning_rate(optimizer_G1, total_iters, max_iters, args.base_lr_G, args.power, True)
            adjust_learning_rate(optimizer_G2, total_iters, max_iters, args.base_lr_G, args.power)
            d_lr = adjust_learning_rate(optimizer_fin_D, total_iters, max_iters, args.base_lr_D, args.power, True)
            adjust_learning_rate(optimizer_Dec, total_iters, max_iters, args.base_lr_Dec, args.power)

            scalar_info_lr = {
                'lr/G1_G2_Dec': g_lr,
                'lr/Dis': d_lr}

            if ask_fake:
                if ask_dataset:
                    rgb_images, th_images, labels, names, fake = batch #KP
                else :
                    rgb_images, th_images, _, labels, names, fake = batch #MF

                if np.random.rand() < args.fake_prob:
                    del th_images
                    th_images = Variable(fake).cuda()
                    for n_idx in range(len(names)):
                        names[n_idx] = names[n_idx] + '_2N'
                else:
                    del fake
                    th_images = Variable(th_images).cuda()
            else:
                if ask_dataset:
                    rgb_images, th_images, labels, names = batch #KP
                else :
                    rgb_images, th_images, _, labels, names = batch #MF



            rgb_images = Variable(rgb_images).cuda()

            ### step 1. train G1 ###
            set_requires_grad([G2, Decoder, fin_D], False)

            # G1(S)<->G2(T)
            mid_pred1 = G1(rgb_images)
            pred1 = Decoder(mid_pred1)
            pred1_seg_loss_G1 = loss_calc(pred1, labels)
            loss_G1 = pred1_seg_loss_G1
            loss_G1.backward()

            scalar_info_G1 = {
                'G1/loss(segmentation loss)': loss_G1.item()}

            LOSS_G1_VALUE += loss_G1.item()

            ### step 2. train G2 ###
            set_requires_grad([G2], True)
            pseudo_label = pred1.detach()

            mid_pred2 = G2(th_images)
            pred2 = Decoder(mid_pred2)

            pseudo_label = pseudo_label.argmax(1)
            pred2_seg_loss_G2 = loss_calc(pred2, pseudo_label)

            # pred2 <-> source label: to fool D2
            fin_D_2 = fin_D(F.softmax(pred2))
            source_label2 = Variable(source_label_tensor.expand(fin_D_2.data.size())).cuda()
            fin_adv_loss_G2 = adv_loss(fin_D_2, source_label2)

            loss_G2 = pred2_seg_loss_G2 + gamma * fin_adv_loss_G2
            loss_G2.backward()

            scalar_info_G2 = {
                'G2/pseudo_seg_loss': pred2_seg_loss_G2.item(),
                'G2/adv_loss': fin_adv_loss_G2.item()}

            LOSS_G2_VALUE_adv += fin_adv_loss_G2.item()
            LOSS_G2_VALUE_seg += pred2_seg_loss_G2.item()

            #### step 3. train Decoder ####
            set_requires_grad([Decoder], True)
            mid_pred1 = mid_pred1.detach()
            # detach prediction from G1 and G2

            pred1 = Decoder(mid_pred1)
            pred1_seg_loss = loss_calc(pred1, labels)

            mid_pred2 = mid_pred2.detach()
            pseudo_label = pred1.detach()
            pseudo_label = pseudo_label.argmax(1)
            pred2 = Decoder(mid_pred2)
            pred2_seg_loss = loss_calc(pred2, pseudo_label)


            # pred2 <-> source label: to fool D2
            fin_D_2 = fin_D(F.softmax(pred2))
            source_label2 = Variable(source_label_tensor.expand(fin_D_2.data.size())).cuda()
            fin_adv_loss = adv_loss(fin_D_2, source_label2)

            loss_decoder = alpha * pred1_seg_loss + beta * pred2_seg_loss + gamma * fin_adv_loss
            loss_decoder.backward()

            scalar_info_decoder = {
                'decoder/pred1_seg_loss': pred1_seg_loss.item(),
                'decoder/pseudo_seg_loss': pred2_seg_loss.item(),
                'decoder/fin_adv_loss': fin_adv_loss.item()}

            LOSS_DECODER_VALUE_pred1 += pred1_seg_loss.item()
            LOSS_DECODER_VALUE_pseudo += pred2_seg_loss.item()
            LOSS_DECODER_VALUE_adv += fin_adv_loss.item()

            #### step 4. train fin_D ####
            set_requires_grad([fin_D], True)
            pred1 = pred1.detach()
            pred2 = pred2.detach()

            fin_D_1 = fin_D(F.softmax(pred1))
            fin_D_2 = fin_D(F.softmax(pred2))

            target_label2 = Variable(target_label_tensor.expand(fin_D_2.data.size())).cuda()

            fin_adv_loss1 = adv_loss(fin_D_1, source_label2)
            fin_adv_loss2 = adv_loss(fin_D_2, target_label2)

            loss_fin_D = (fin_adv_loss1 + fin_adv_loss2) / 2
            loss_fin_D.backward()

            scalar_info_fin_D = {
                'fin_D/loss': loss_fin_D.item(),
                'fin_D/loss_fin_adv1': fin_adv_loss1.item(),
                'fin_D/loss_fin_adv2': fin_adv_loss2.item()
            }

            LOSS_FIN_D_VALUE += loss_fin_D.item()

            optimizer_G1.step()
            optimizer_G2.step()
            optimizer_Dec.step()
            optimizer_fin_D.step()

            if args.tensorboard:
                for key, val in scalar_info_lr.items():
                    writer.add_scalar(key, val, total_iters)
                for key, val in scalar_info_G1.items():
                    writer.add_scalar(key, val, total_iters)
                for key, val in scalar_info_G2.items():
                    writer.add_scalar(key, val, total_iters)
                for key, val in scalar_info_fin_D.items():
                    writer.add_scalar(key, val, total_iters)
                for key, val in scalar_info_decoder.items():
                    writer.add_scalar(key, val, total_iters)

            content = 'iter = {0:6d}/{1:6d}, G1 = {2:.6f} G2 = {3:.6f} G2_adv ={4:.6f} dec_pred1 = {5:.6f} ' \
                      'dec_pred2 = {6:.6f} dec_adv = {7:.6f} fin_D1 = {8:.6f} fin_D2 ={9:.6f}'.format(
                total_iters, max_iters,
                loss_G1.item(),
                pred2_seg_loss_G2.item(), fin_adv_loss_G2.item(),
                pred1_seg_loss.item(), pred2_seg_loss.item(),fin_adv_loss.item(),
                fin_adv_loss1.item(), fin_adv_loss2.item())
            print(content)

            total_iters += args.train_batch
            with open(record_filename, 'a') as txtfile:
                txtfile.write(content + '\n')
                txtfile.write('g_lr:{0:.10f}, d_lr:{1:.10f}\n'.format(g_lr, d_lr))
            print('time taken: %.3f' % (time.time() - iter_start_time))

            if i % 100 == 0 :
                TL_DA_visualize(rgb_images, th_images, labels, pred1, pred2, names, check_dir, epoch=epoch)
                print('visualize predictions ...')
            if total_iters == 0:
                print('save test model ...')
                torch.save(model.state_dict(), os.path.join(check_dir, model_name + '_' + 'test' + '.pth'))

        content2 = 'epoch{0:4d}/{1:4d}, total loss for 1 epoch = G1 = {2:.6f} G2 = {3:.6f} G2_adv = {4:.6f} dec_pred1 = {5:.6f}' \
                   ' dec_pred2 = {6:.6f} dec_adv = {7:.6f} fin_D = {8:.6f} '.format(epoch, args.max_epoch,
                                                                                    LOSS_G1_VALUE / len(trainloader),
                                                                                    LOSS_G2_VALUE_seg / len(
                                                                                        trainloader),
                                                                                    LOSS_G2_VALUE_adv / len(
                                                                                        trainloader),
                                                                                    LOSS_DECODER_VALUE_pred1 / len(
                                                                                        trainloader),
                                                                                    LOSS_DECODER_VALUE_pseudo / len(
                                                                                        trainloader),
                                                                                    LOSS_DECODER_VALUE_adv / len(
                                                                                        trainloader),
                                                                                    LOSS_FIN_D_VALUE / len(trainloader))
        with open(record_epoch_filename, 'a') as epochrecord:
            epochrecord.write(content2 + '\n')

        print('save model ...')
        torch.save(model.state_dict(), os.path.join(check_dir, model_name + '_' + str(epoch) + '.pth'))
        torch.save(optimizer_G1.state_dict(), os.path.join(check_dir, model_name + '_' + str(epoch) + '_G1.optim'))
        torch.save(optimizer_G2.state_dict(), os.path.join(check_dir, model_name + '_' + str(epoch) + '_G2.optim'))
        torch.save(optimizer_Dec.state_dict(), os.path.join(check_dir, model_name + '_' + str(epoch) + '_Dec.optim'))
        torch.save(optimizer_fin_D.state_dict(),
                   os.path.join(check_dir, model_name + '_' + str(epoch) + '_fin_D.optim'))
        torch.save(model.state_dict(), os.path.join(check_dir, model_name + '_latest_model.pth'))
        print('time taken: %.3f sec per epoch' % (time.time() - epoch_start_time))

    if args.tensorboard:
        writer.close()
