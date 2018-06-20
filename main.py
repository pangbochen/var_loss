import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.stats import pearsonr, spearmanr

from model import NSM, LM
from dataset import FeatureVarDataset, FeatureDataset
from loss import VarLoss
from visualize import Visualizer

# init visualizer
vis = Visualizer('var_loss')

n_feature = 42
data_FN = 'data.csv'
batch_size = 256
lr = 0.01
epochs = 300

train_loader  = torch.utils.data.DataLoader(
    FeatureDataset(data_FN=data_FN,type='train'),
    batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    FeatureDataset(data_FN=data_FN,type='test'),
    batch_size=batch_size, shuffle=True
)

model = LM(n_feature=n_feature)

optimizer = optim.Adagrad(model.parameters(),lr=lr)

# acc, for report use
best_loss = 1000

for epoch in range(epochs):
    print(epoch)
    model.train()

    for batch_index, (data, target) in enumerate(train_loader):
        # for debug use
        # if batch_index > 10:
        #     break
        index_target = target.clone()


        data, target = Variable(data), Variable(target)

        data, target = data.float(), target.float()

        # zero_grad
        optimizer.zero_grad()
        # get output from mlp model
        output = model(data)

        # for debug
        # if batch_index == 2:
        #     print(data)
        #     print(target)
        #     print(output)

        # use cross_entropy
        # cross_loss_matrix = F.mse_loss(target, output, reduce=False)
        # loss = torch.std(cross_loss_matrix) + cross_loss_matrix.mean()

        # use l1 loss to smooth
        # loss_matrix = F.l1_loss(target, output, reduce=False)
        # loss = torch.std(loss_matrix) + loss_matrix.mean()

        # use smooth_l1_loss to smooth the loss function
        loss_matrix = F.smooth_l1_loss(output, target, reduce=False)
        loss = torch.std(loss_matrix) + loss_matrix.mean()

        # loss_matrix
        loss.backward()

        optimizer.step()
        vis.plot('mixed_loss', min(loss.data, 10))
        vis.plot('origin loss', min(loss_matrix.mean().data, 10))
        vis.plot('std', torch.std(loss_matrix).data)

    # 评估本轮的ic值
    model.eval()
    y_pred = torch.Tensor()
    y_true = torch.Tensor()

    for batch_index, (data, target) in enumerate(test_loader):
        data, target = Variable(data), Variable(target)
        data, target = data.float(), target.float()
        output = model(data)
        # update y_pred and y_true
        y_pred = torch.cat([y_pred, output])
        y_true = torch.cat([y_true, target])

    y_true = y_true.data.numpy()
    y_pred = y_pred.data.numpy()

    rank_ic = spearmanr(y_pred, y_true)[0]
    normal_ic = pearsonr(y_pred, y_true)[0]
    # plot ic
    print(rank_ic)
    print(normal_ic)
    vis.plot('rank ic', rank_ic)
    vis.plot('normal ic', normal_ic)
    model.train()