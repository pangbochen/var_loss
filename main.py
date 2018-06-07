


import torch




from model import NSM
from dataset import




n_feature = 172
data_FN = 'dataset.pkl'

model = NSM(n_feature=n_feature)

train_loader  = torch.util