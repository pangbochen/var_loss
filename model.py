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

# the niave multi-factor linear model
# regression task
# predict return directly the return value
class LRM(nn.Module):

    def __init__(self, n_feature):
        super(LRM, self).__init__()
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

# classification task
class LCM(nn.Module):
    def __init__(self, n_feature, n_label):
        super(LCM, self).__init__()
        # n_feature:使用的因子的个数
        self.linear1 = nn.Linear(in_features=n_feature, out_features=n_feature*2)

        self.linear2 = nn.Linear(in_features=n_feature*2, out_features=n_label)

    def forward(self, x):
        '''Parameters
        x : input_size is (batch_size,  n_feature)
        output: pred return for each record (batch_size) return the actual label for this record
        '''
        x = self.linear1(x)
        x = self.linear2(x)
        # there are two ways to handle this problem
        # argmax or the distribution way
        # x = torch.argmax(x, dim=-1)
        return x