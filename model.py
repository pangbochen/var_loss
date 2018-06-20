import torch
import torch.nn as nn
import torch.nn.functional as F

class NSM(nn.Module):

    def __init__(self, n_feature):
        super(NSM, self).__init__()
        # n_feature:使用的因子的个数
        self.linear1 = nn.Linear(in_features=n_feature, out_features=n_feature*2)

        self.linear2 = nn.Linear(in_features=n_feature*2, out_features=1)

    def forward(self, f_g_hist):
        '''Parameters
        f_g_hist : input_size is (batch_size, group_size, n_feature)

        output: pred return for each record (batch_size, group_size, 1)
        '''
        x = self.linear1(f_g_hist)
        x = self.linear1(x)
        x = x.squeeze(-1)
        return x


class LM(nn.Module):

    def __init__(self, n_feature):
        super(LM, self).__init__()
        # n_feature:使用的因子的个数
        self.linear1 = nn.Linear(in_features=n_feature, out_features=n_feature*2)

        self.linear2 = nn.Linear(in_features=n_feature*2, out_features=1)

    def forward(self, x):
        '''Parameters
        x : input_size is (batch_size,  n_feature)

        output: pred return for each record (batch_size, 1)
        '''
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.squeeze(-1)
        return x
