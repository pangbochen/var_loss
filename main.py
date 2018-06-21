import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.stats import pearsonr, spearmanr

from model import NSM, LCM, LRM
from dataset import FeatureVarDataset, FeatureDataset
from loss import VarLoss
from visualize import Visualizer

batch_size = 256
lr = 0.01
epochs = 300

# For data file, classifcatoin and the regression problems are different
# we need to preprocess the data to generate the suitable label int the preprocess part
# use label_column to handle classification and regression

task_type = 'classification' # or regression
n_feature = 43
data_FN = 'data.csv'

# n_feature = 157
# data_FN = 'D:/data/data.csv'

if task_type == 'classification':
    label_column = 'label_3'  # set the label column of the dataframe file
    n_label = 3
    model = LCM(n_feature=n_feature, n_label=n_label)
elif task_type == 'regression':
    label_column = 'return'  # set the label column of the dataframe file
    model = LRM(n_feature=n_feature)


train_loader  = torch.utils.data.DataLoader(
    FeatureDataset(data_FN=data_FN,type='train', label_column=label_column),
    batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    FeatureDataset(data_FN=data_FN,type='test', label_column=label_column),
    batch_size=batch_size, shuffle=True
)


optimizer = optim.Adagrad(model.parameters(),lr=lr)

# acc, for report use
best_loss = 1000

plot_batch = 100

print('begin...')

# init visualizer
vis = Visualizer('var_loss')

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

        '''
        Part II
        classification part
        For example, 
            3_class_classification 
            模型直接得到record在不同类别上的概率分布，再得到具体的，loss使用分类问题的loss，合理的model评测指标时AUC的分类指标，无法得到rank_ic指标值
        '''

        # use cross_entropy
        print(target.shape)
        print(output.shape)
        target = target.long()
        loss_matrix = F.cross_entropy(output, target, reduce=False)
        loss = torch.std(loss_matrix) + loss_matrix.mean()
        print(loss.requires_grad)
        '''
        Part I
        regression part
        
        建立多因子模型对股票进行建模
        
        直接预测股票的收益
        
        模型的loss即为真实收益和预测收益之间的差值
        '''

        # use l1 loss to smooth
        # loss_matrix = F.l1_loss(target, output, reduce=False)
        # loss = torch.std(loss_matrix) + loss_matrix.mean()

        # use smooth_l1_loss to smooth the loss function
        # loss_matrix = F.smooth_l1_loss(output, target, reduce=False)
        # loss = torch.std(loss_matrix) + loss_matrix.mean()

        # use smooth_l1_loss
        # use sum(abs(loss-mean)) to evaluate the volatility of the model prediction
        # use l1 loss to make
        # the problem is
        # loss_matrix = F.smooth_l1_loss(output, target, reduce=False)
        # loss = torch.abs(loss_matrix - loss_matrix.mean()).mean() + loss_matrix.mean()

        # loss_matrix
        loss.backward()

        optimizer.step()
        # plot loss for the epoch for every plot_batch
        if batch_index % plot_batch == 0:
            vis.plot('mixed_loss', min(loss.data, 10))
            vis.plot('origin_loss', min(loss_matrix.mean().data, 10))
            vis.plot('std', torch.std(loss_matrix).data)

    if task_type == 'regression':
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