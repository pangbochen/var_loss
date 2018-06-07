# define own Dataset
# two
# __getitem__
# __len__

import torch
from torch.utils import data

import os
import pandas as pd
import numpy as np
import pickle

class FeatureVarDataset(data.Dataset):
    def __init__(self, data_FN='dataset.pkl', group_size=64, type='train'):
        self.group_size = group_size
        self.data_FN = data_FN
        self.feature, self.label = self.precess(self.data_FN, type)


    def precess(self, data_FN, type):
        # 由于要使用var作为优化目标，将原始的数据组成group，优化group内mse varience
        with open(data_FN, 'rb') as f:
            data_df = pickle.load(f)

        data_df = self.data_precess(data_df)

        feature_column = list(data_df.columns[2:-3])
        label_column = 'FCHG1'

        num_records = data_df.shape[0]
        index_slice = int(num_records*0.8)

        if type == 'train':
            feature = data_df[feature_column].values[:index_slice]
            label = data_df[label_column].values[:index_slice]
        elif type == 'test':
            feature = data_df[feature_column].values[index_slice:]
            label = data_df[label_column].values[index_slice:]

        return feature, label

    def data_precess(self, data_df):
        # TODO should not in dataset
        data_df = data_df.dropna()
        # 按照时间排序
        # data_df = data_df.reset_index().sort_values(by=['datetime'])

        return data_df


    def __len__(self):
        return int(len(self.feature) // self.group_size)

    def __getitem__(self, index):
        # 返回下标为index的group数据， shape is (group_size, feature_shape)
        data = self.feature[index*self.group_size:(index+1)*self.group_size]
        label = self.label[index*self.group_size:(index+1)*self.group_size]
        return data, label

