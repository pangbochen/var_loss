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
