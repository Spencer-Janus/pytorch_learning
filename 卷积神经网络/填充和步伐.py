# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 14:20:42 2022

@author: Janus_yu
"""
import torch
from torch import nn

def comp_conv2d(conv2d,X):
    X=X.reshape((1,1)+X.shape)
    Y=conv2d(X)
    return Y.reshape(Y.shape[2:])
conv2d=nn.Conv2d(1, 1, kernel_size=3,padding=1)
X=torch.rand(size=(8,8))
print(comp_conv2d(conv2d, X).shape)


conv2d=nn.Conv2d(1, 1, kernel_size=(5,3),padding=(2,1))
print(comp_conv2d(conv2d, X).shape)



conv2d=nn.Conv2d(1, 1, kernel_size=3,padding=1,stride=2)
print(comp_conv2d(conv2d, X).shape)


conv2d=nn.Conv2d(1, 1, kernel_size=(3,5),padding=(0,1),stride=(3,4))
print(comp_conv2d(conv2d, X).shape)



'''
1.填充：我们常常丢失边缘像素。由于我们通常使用小卷积核，因此对于任何单个卷积，我们可能只会丢失几个像素。但随着我们应用许多连续卷积层，累积丢失的像素数就多了。
      为了解决这个问题在图像的边界元素填充0.可以让神经网络更深
      控制输出的减少量
通常填充ph=kh-1 pw=kw-1
kh奇数：上下填充ph/2
kh为偶数：上册ph/2取上，下侧ph/2取下

2.步幅：行/列的滑动步长，成倍减少输出形状

3.三个超参数的重要程度大小：
kernel的大小最关键一般取奇数，上下填充一样 
填充=kernel-1(上下都算) 
步幅取1，若图片太大，复杂度较大时候步幅取2，

4.一般来说不会手写神经网络，除非输入非常非常不一样，如图片为20*1000，
  一般直接在经典架构上进行调整，resnet 
  
5.
'''