# define own Dataset
# two
# __getitem__
# __len__

from torch.utils import data

import os
import pickle
import pandas as pd

class FeatureVarDataset(data.Dataset):
    def __init__(self, data_FN='dataset.pkl', group_size=64, type='train'):
        self.group_size = group_size
        self.data_FN = data_FN
        self.feature, self.label = self.precess(self.data_FN, type)


    def precess(self, data_FN, type):
        # 由于要使用var作为优化目标，将原始的数据组成group，优化group内mse varience
        data_df = pd.read_csv(data_FN, index_col=0)

        data_df = self.data_precess(data_df)

        feature_column = list(data_df.columns[2:-3])
        label_column = 'return'

        num_records = data_df.shape[0]
        index_slice = int(num_records*0.8)

        if type == 'train':
            feature = data_df[feature_column].values[:index_slice]
            label = data_df[label_column].values[:index_slice]
        elif type == 'test':
            feature = data_df[feature_column].values[index_slice:]
            label = data_df[label_column].values[index_slice:]
        else:
            raise NotImplementedError

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
        # 此外还有累进striding的方法，这里实现的非常简单
        return data, label

class FeatureDataset(data.Dataset):
    def __init__(self, data_FN='dataset.pkl', type='train'):
        self.data_FN = data_FN
        self.feature, self.label = self.precess(self.data_FN, type)


    def precess(self, data_FN, type):
        # 由于要使用var作为优化目标，将原始的数据组成group，优化group内mse varience
        # 使用NN中batch_size的概念，优化长度为batch_szie中的loss值
        data_df = pd.read_csv(data_FN, dtype={'stock_id':str})

        data_df = self.data_precess(data_df)

        feature_column = list(data_df.columns[2:-3])
        label_column = 'return'

        num_records = data_df.shape[0]
        index_slice = int(num_records*0.8)

        if type == 'train':
            feature = data_df[feature_column].values[:index_slice]
            label = data_df[label_column].values[:index_slice]
        elif type == 'test':
            feature = data_df[feature_column].values[index_slice:]
            label = data_df[label_column].values[index_slice:]
        else:
            raise NotImplementedError

        return feature, label

    def data_precess(self, data_df):
        # TODO
        # should not do this in dataset part
        # drop na
        data_df = data_df.dropna()
        # 进行百分度的排序操作
        fendu_process = False
        if fendu_process:
            high_quantile = 0.95
            low_quantile = 0.05

            def func(x, high=1, low=1):
                if x > high:
                    return high
                elif x < low:
                    return low
                else:
                    return x

            feature_list = data_df.columns
            feature_list = feature_list[2:]
            for feature in feature_list:
                high = data_df[feature].quantile(high_quantile)
                low = data_df[feature].quantile(low_quantile)
                data_df[feature] = data_df[feature].apply(lambda x: func(x, high, low))
            # 对机制进行处理
        # 按照时间排序
        # data_df = data_df.reset_index().sort_values(by=['datetime'])
        return data_df

    def __len__(self):
        return int(len(self.feature) )

    def __getitem__(self, index):
        # 返回下标为index的group数据， shape is (group_size, feature_shape)
        data = self.feature[index]
        label = self.label[index]
        # 此外还有累进striding的方法，这里实现的非常简单
        return data, label

if __name__ == '__main__':
    data_FN = 'C:/Users/pangbochen/Documents/finance/Efund/DT_RF/data.csv'
    print(os.path.exists(data_FN))
    dataset = FeatureVarDataset(data_FN)
    data, label = dataset[0]