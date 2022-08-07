# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 14:51:19 2022

@author: Janus_yu
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
train_data = pd.read_csv('D:\\torch_leran\\data\\kaggle_house_pred_train.csv')
test_data = pd.read_csv('D:\\torch_leran\\data\\kaggle_house_pred_test.csv')
# print(train_data.shape)
# print(test_data.shape)
# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))


numeric_features=all_features.dtypes[all_features.dtypes!='object'].index #object 文字，选出数值列， 
all_features[numeric_features] = all_features[numeric_features].apply(
lambda x: (x - x.mean()) / (x.std()))  #数值行feature normalization

# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)  #将NA填充为0

all_features = pd.get_dummies(all_features, dummy_na=True)#独热编码
print(all_features.shape)
#，通过values属性，我们可以从pandas格式中提取NumPy格式，并将其转换为张量表⽰⽤于训练。
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)

test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)

train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
#训练
loss = nn.MSELoss()#损失函数

in_features = train_features.shape[1] #features的数量

net = nn.Sequential(nn.Linear(in_features,1))#模型

def log_rmse(net, features, labels):
# 为了在取对数时进⼀步稳定该值，将⼩于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))#1-无穷大
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
    torch.log(labels)))
    return rmse.item() #返回高精度值

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这⾥使⽤的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate,
    weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
        

#k折交叉验证
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid
#k折训练 
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs], legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
        f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 6, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
f'平均验证log rmse: {float(valid_l):f}')    

#调好参数之后在完整数据上训练一次
























'''
1 iloc[]函数作用
iloc[]函数，属于pandas库，全称为index location，即对数据进行位置索引，从而在数据表中提取出相应的数据。

2、concat()
pd.concat(objs, axis=0, join=‘outer’, join_axes=None, ignore_index=False,
keys=None, levels=None, names=None, verify_integrity=False,
copy=True)
常用参数：
axis：{0,1，…}，默认为0，也就是打竖，上下拼接。
join：{‘inner’，‘outer’}，默认为“outer”。outer为并集、inner为交集。
join_axes：Index对象列表。

3.pandas默认的数字类型是 int64 和 float64，文字类型是 object。

4.pd.get_dummies(all_features, dummy_na=True)  使得字符串，独热编码，并对未采样的数据(NA)创建指示符特征
all_features为readcsv返回的文件
5.all_features.dtypes所有列的类型.

6.clamp（）函数：
    参考资料：https://blog.csdn.net/jacke121/article/details/85270621
    1）函数功能：
          clamp（）函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量。
    2）参数列表：
torch.clamp(input, min, max, out=None) → Tensor
          input：输入张量；
          min：限制范围下限；
          max：限制范围上限；
7.关于为什么直接对l求导，l不是向量嘛？
很多的loss函数都有size_average和reduce两个布尔类型的参数，因为一般损失函数都是直接计算batch的数据，因此返回的loss结果都是维度为(batch_size,)的向量。

1）如果reduce=False,那么size_average参数失效，直接返回向量形式的loss

2)如果redcue=true,那么loss返回的是标量。

   2.a: if size_average=True, 返回loss.mean();#就是平均数

   2.b: if size_average=False,返回loss.sum()

注意：默认情况下，reduce=true,size_average=true
8，torch.cat((A,B),0)#按维数0（行）拼接 A,B按照维度0拼接

9.cross_entropy自带 softmax操作

10.Adam优化算法是什么？


'''
