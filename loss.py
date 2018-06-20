import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable

class VarLoss(Function):

    def __init__(self):
        super(VarLoss, self).__init__()
        return

    def forward(self, pred, target):
        # input : (batch_size, group_size, 1)
        loss = pred - target
        loss = loss.pow(2)
        loss = loss.std(dim=1).squeeze()
        return loss

    def backward(ctx, *grad_outputs):
        return None


def var_loss_01(target, output):
    cross_loss_matrix = F.mse_loss(target, output, reduce=False)
    loss = torch.std(cross_loss_matrix) + cross_loss_matrix.mean()
    return loss


def var_loss_02(target, output):
    '''Parameters
    target : (batch_size) prediction for the label

    abs l1
    var
    '''
    loss = torch.abs(target-output)
