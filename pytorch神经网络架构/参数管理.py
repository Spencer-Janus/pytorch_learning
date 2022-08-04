# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:04:19 2022

@author: Janus_yu
"""
import torch
from torch import nn
net=nn.Sequential(nn.Linear(4,8),nn.Relu(),nn.Linear(8,1))
X=torch.rand(size=(2,4))
net(X)

print(net[2].state_dict())#最后一层参数
print(net[2].bias)#他还有梯度
print(net[2].bias.data)
print(net[2].weight.grad)
#1.一次性访问所有参数
print(*[(name,param.shape)for name,param in net[0].named_parameters()])
print(*[(name,param.shape)for name,param in net.named_parameters()])
#去掉星号*会发生什么？？？


#如何从嵌套块收集参数
def block():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,4),nn.ReLU())

def block2():
    net=nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}', block())
    return net
rgnet=nn.Sequential(block2(),nn.Linear(4,1))
rgnet(X)
print(rgnet)#可以查看网络大概的样子

#2.内置的初始化参数
def init_normal(m):
    if type(m)==nn.Linear:#如果是全连接层
        nn.init.normal_(m.weight,mean=0,std=0.01)#init里包含替换函数
        nn.init.zeros_(m.bias)
net.apply(init_normal)
print(net[0].weight.data[0],net[0].bias.data[0])

def init_constant(m):
    if type(m)==nn.Linear:
        nn.init.constant_(m.weight,1)  
        nn.init.zeros_(m.bias)
net.apply(init_constant)
print(net[0].weight.data[0],net[0].bias.data[0])
def xavier(m):
    if type(m)==nn.Linear:
        nn.init.xavier_uniform_(m.weight)
net[0].apply(xavier)
print(net[0].weight.data[0])

#3.自定义初始化
def my_init(m):
    if type(m)==nn.Linear:
        print(
            'init',
            *[(name,param.shape)for name,param in m.named_parameters()][0])
        nn.init.uniform(m.weight,-10,10)
        m.weight.data*=m.weight.data.abs()>=5 #>=的优先级高于*=
net.apply(my_init)
# print(net[0].weight[:2])
#直接赋值
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
#4.参数绑定
shared=nn.Linear(8,8)
net=nn.Sequential(nn.Linear(4,8),nn.Relu(),shared,nn.ReLU(),shared,nn.ReLU(),nn.Linear(8,1))
net(X)
print(net[2].weight.data[0]==net[4].weight.data[0])
net[2].weight.data[0,0]==100
print(net[2].weight.data[0]==net[4].weight.data[0])

#2 4 是shared这个网络 它们共享参数




      
'''
1.apply给你一个方式loop整个网络。不仅可以初始化参数，也可以干别的事情

2.weight全一样，一层相当只有一个神经元了？

3.xavier初始化https://blog.csdn.net/shuzfan/article/details/51338178?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-51338178-blog-89556165.pc_relevant_sortByAnswer&spm=1001.2101.3001.4242.1&utm_relevant_index=3
'''





