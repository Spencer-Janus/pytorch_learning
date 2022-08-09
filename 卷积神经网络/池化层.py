# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 19:57:49 2022

@author: Janus_yu
"""
#池化层的手动实现
import torch
from torch import nn
from d2l import torch as d2l
def pool2d(X,pool_size,mode='max'):
    p_h,p_w=pool_size
    Y=torch.zeros((X.shape[0]-p_h+1,X.shape[1]-p_w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode=="max":
                Y[i,j]=X[i:i+p_h,j:j+p_w].max()
            elif mode=='avg':
                Y[i,j]=X[i:i+p_h,j:j+p_w].mean()
    return Y
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(pool2d(X, (2, 2)))
print(pool2d(X, (2, 2)),'avg')


#简单实现步伐和填充
X=torch.arange(16,dtype=torch.float32).reshape((1,1,4,4))

pool2d=nn.MaxPool2d(3)
print(pool2d(X))

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)

#池化层在每个输入通道上单独运算
X=torch.cat((X,X+1),1)
pool2d=nn.MaxPool2d(3,padding=1,stride=2)
print(pool2d(X))







'''
1.池化层容许输入的稍微偏移，缓解卷积层对位置的敏感性，也有填充和步幅

2.对每个通道做池化，没有可学习的参数，输入通道数=输出通道数

3.最大池化：每个窗口最强的模式信号，平均池化：平均

4.默认情况下，深度学习框架中的步幅与池化窗口的大小相同。因此，如果我们形状为(3, 3)的池化窗
口，那么默认情况下，我们得到的步幅形状为(3, 3)。
'''