# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 15:51:54 2022

@author: Janus_yu
"""
import torch
from d2l import torch as d2l
#手写法
# def corr2d_multi_in(X,K):
#     return sum(d2l.corr2d(x,k)for x, k in zip(X,K))
# X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
#                   [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
# K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
# # print(corr2d_multi_in(X, K))
# def corr2d_multi_in_out(X,K):
#     return torch.stack([corr2d_multi_in(X, K) for k in K],0)

def corr2d_multi_in_out_1x1(X,K):
    c_i,h,w=X.shape
    c_o=K.shape[0]
    X=X.reshape((c_i,h*w))
    K=K.reshape((c_o,c_i))
    Y=torch.matmul(K,X)
    print(Y)
    return Y.reshape((c_o,h,w))
#简洁实现

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

corr2d_multi_in_out_1x1(X, K)













'''
1.每个输入通道，卷积核参数不一样 每个卷积核对应一个偏差B

2.每个输出通道识别某个特定的模式

3.1x1卷积层 卷积核为1x1 融合不同通道的信息，相当于一个全连接层输入为((nhnw)*1),权重coci 图见21卷积层的多输入多输出通道

4.zip()是对最外边的维度做zip

5.torch.stack沿一个新维度对输入张量序列进行连接()序列中所有张量应为相同形状；stack 函数返回的结果会新增一个维度，而stack（）函数指定的dim参数，就是新增维度的（下标）位置。
dim：新增维度的（下标）位置，当dim = -1时默认最后一个维度；

6.paddle0越多 对性能影响不大(计算性能、模型性能)

7.卷积核bias对结果的影响不大

8.卷积核的参数值是学出来的 输入通道输入决定，输出通道卷积核维度0决定

'''