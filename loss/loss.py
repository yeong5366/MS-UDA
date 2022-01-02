import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

def loss_calc(pred, label, gpu=None, ignore_coarse_label=None):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    if gpu is None:
        label = Variable(label.long()).cuda()
        criterion = CrossEntropy2d(ignore_coarse_label=ignore_coarse_label).cuda()
    else:
        label = Variable(label.long()).cuda(gpu)
        criterion = CrossEntropy2d(ignore_coarse_label=ignore_coarse_label).cuda(gpu)

    return criterion(pred, label)

class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255, ignore_coarse_label = None):
        super(CrossEntropy2d,self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label
        self.ignore_coarse_label = ignore_coarse_label

    def forward(self, predict, target, weight=None):

        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        if self.ignore_coarse_label is None:
            target_mask = (target >= 0) * (target != self.ignore_label)
        else:
            target_mask = (target >= 0) * (target != self.ignore_label) * (target < self.ignore_coarse_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous() #BxCxHxW-> BxHxWxC
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, reduction='mean') #size_average=self.size_average)
        return loss

