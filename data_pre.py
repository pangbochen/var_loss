import pickle



data_FN = './raw/dataset.pkl'

with open(data_FN, 'rb') as f:
    data_df = pickle.load(f)

data_df = data_df.dropna()
# 按照时间排序
data_df = data_df.reset_index().sort_values(by=['datetime'])

output_FN = 'dataset.pkl'
with open(output_FN, 'wb') as f:
    pickle.dump(data_df, f)