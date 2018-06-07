import torch
import torch.optim as optim
from torch.autograd import Variable


from model import NSM
from dataset import FeatureVarDataset
from loss import VarLoss



n_feature = 172
data_FN = 'dataset.pkl'
batch_size = 32
lr = 0.01
epochs = 300

train_loader  = torch.utils.data.DataLoader(
    FeatureVarDataset(data_FN=data_FN,group_size=64,type='train'),
    batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    FeatureVarDataset(data_FN=data_FN,group_size=64,type='test'),
    batch_size=batch_size, shuffle=True
)

model = NSM(n_feature=n_feature)

optimizer = optim.Adagrad(model.parameters(),lr=lr)

# acc, for report use
best_loss = 1000

for epoch in range(epochs):
    model.train()

    for batch_index, (data, target) in enumerate(train_loader):
        index_target = target.clone()

        data, target = Variable(data), Variable(target)

        # zero_grad
        optimizer.zero_grad()
        # get output from mlp model
        output = model(data)

        # use cross_entropy
        loss = VarLoss(output, target)
        loss.backward()
        optimizer.step()

        print(loss)