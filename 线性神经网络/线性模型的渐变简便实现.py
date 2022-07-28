# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:08:14 2022

@author: Janus_yu
"""

import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

#构造人造数据集 w=[2,-3.4] b=4.2
def synthetic_data(w,b,num_examples):
    x=torch.normal(0,1,(num_examples,len(w)),dtype=torch.float32)
    y=torch.matmul(x,w)+b
    y+=torch.normal(0,0.01,y.shape) #加入噪音
    return x,y.reshape(-1,1)
#-1表示行数由torch自动计算
true_w=torch.tensor([2,-3.4],dtype=torch.float32)
true_b=4.2
features,labels=synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):
#构造1个PyTorch数据迭代器
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
batch_size = 10
data_iter = load_array((features, labels), batch_size)
#使用框架预定义好的层
net=nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)
#损失函数 均方误差
loss=nn.MSELoss()

#实例化SGD 优化算法 ：小批量随机梯度下降
trainer=torch.optim.SGD(net.parameters(),lr=0.03)

num_epochs=3
for epoch in range(num_epochs):
    for x,y in data_iter:
        l=loss(net(x),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l=loss(net(features),labels)
    print(f'epoch {epoch + 1}, loss {l:f},sec')
print(net[0].weight.data)
print(net[0].bias.data)
'''
dataloader 按照batch_size打乱加载数据
TensorDataset(*data_arrays)
*解包

accumulator中*为收集元素
**为收集字典
zip函数
>>> a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> zipped = zip(a,b)     # 打包为元组的列表
[(1, 4), (2, 5), (3, 6)]
'''






