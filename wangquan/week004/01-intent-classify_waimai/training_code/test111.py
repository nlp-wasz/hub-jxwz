import pandas as pd


# 加载和预处理数据
# dataset_df = pd.read_csv('assets/dataset/dataset.csv', sep='\t', header=None)
dataset_df = pd.read_csv('../assets/dataset/waimai_10k.csv', sep=',', header=None, skiprows=1)
# print("dataset_df[1].values", dataset_df[0].values)
print("labels", list(set(dataset_df[0].values)))