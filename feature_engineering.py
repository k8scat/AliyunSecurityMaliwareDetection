# -*- coding: utf-8 -*-

"""
@author: hsowan <hsowan.me@gmail.com>
@date: 2019/11/22

Refer: https://xz.aliyun.com/t/3704

Feature Engineering:

First, group by file_id,
then, sort all api by tid and index (dataset is already sorted),
finally, get train_data (file_id + label + sorted_apis) and test_data (file_id + sorted_apis)

"""

import pandas as pd
import pickle


# 根据file_id进行分组，再根据tid和index对api进行排序
def read_train(df):
    labels = []
    apis = []
    file_ids = []
    group_by_file_id = df.groupby('file_id')
    for file_id, file_group in group_by_file_id:
        file_ids.append(file_id)
        file_label = file_group['label'].values[0]
        # file_group.sort_values(['tid', 'index'], ascending=True, inplace=True)
        api_sequence = ' '.join(file_group['api'])
        labels.append(file_label)
        apis.append(api_sequence)
    return pd.DataFrame([labels, apis], index=['label', 'apis'], columns=file_ids).transpose()


def read_test(df):
    apis = []
    file_ids = []
    group_by_file_id = df.groupby('file_id')
    for file_id, file_group in group_by_file_id:
        file_ids.append(file_id)
        # file_group.sort_values(['tid', 'index'], ascending=True, inplace=True)
        api_sequence = ' '.join(file_group['api'])
        apis.append(api_sequence)
    return pd.DataFrame([apis], index=['apis'], columns=file_ids).transpose()


# 读取训练集和测试集
train = pd.read_csv('./datasets/security_train.csv')
test = pd.read_csv('./datasets/security_test.csv')

train_data = read_train(train)
test_data = read_test(test)

# 对象持久化
with open('data.pkl', 'wb') as f:
    pickle.dump(train_data, f)
    pickle.dump(test_data, f)
