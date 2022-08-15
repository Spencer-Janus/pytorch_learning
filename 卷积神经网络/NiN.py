# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 15:17:21 2022

@author: Janus_yu
"""
import torch
from torch import nn
from d2l import torch as d2l
def nin_block(in_channel,out_channel,kernel_size,strides,padding):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size,strides,padding),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=1),
        nn.ReLU())
        
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成⼆维的输出，其形状为(批量⼤⼩,10)
    nn.Flatten())       
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)       
# lr, num_epochs, batch_size = 0.1, 10, 128
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
# d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())           

'''
1.如何理解1x1卷积增加每个像素的非线性性
  实质:对每个像素不同通道做了个全连接层,通道融合
2.全连接层参数太多，开销大，容易过拟合
3.NiN块一个卷积层跟了两个1x1卷积（全连接层）
4.nn.Flatten())  从dim=1维度一直展开到最后
'''