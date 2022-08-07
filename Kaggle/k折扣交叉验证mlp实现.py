# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:25:59 2022

@author: Janus_yu
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F



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
#模型
in_features = train_features.shape[1] #features的数量
class MLP(nn.Module):
    def __init__(self):
        super().__init__() #父类初始化
        self.hidden=nn.Linear(in_features,256)
        self.hidden2=nn.Linear(256,128)
        self.out=nn.Linear(128,1)
    def forward(self,X):
        model=self.hidden(X)
        model=F.relu(model)
        model=self.hidden2(model)
        model=F.relu(model)
        return self.out(model)
net=MLP()
def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.normal(m.weight,std=0.01)
net.apply(init_weights);   
#损失函数
loss= nn.MSELoss()
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
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.3, 0, 64
# train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,weight_decay, batch_size)
# print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
# f'平均验证log rmse: {float(valid_l):f}')    
def train_and_pred(train_features, test_features, train_labels, test_data,
num_epochs, lr, weight_decay, batch_size):
    net = MLP()
    train_ls, _ = train(net, train_features, train_labels, None, None,
    num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将⽹络应⽤于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
train_and_pred(train_features, test_features, train_labels, test_data,num_epochs, lr, weight_decay, batch_size)
'''
调参，敏感性

'''
