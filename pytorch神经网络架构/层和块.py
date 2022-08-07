# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:17:49 2022

@author: Janus_yu
"""

import torch
from torch import nn
from torch.nn import functional as F
net=nn.Sequential(nn.Linear(20, 256),nn.ReLU(),nn.Linear(256,10))
#在这里nn.Relu()是构造了一个类对象，而非函数调用
X=torch.rand(2,20)
#1.nn.moudle法
class MLP(nn.Module):
    def __init__(self):
        super().__init__() #父类初始化
        self.hidden=nn.Linear(20,256)
        self.out=nn.Lienar(256,10)
    def forward(self,X):
        return self.out(F.relu(self.hidden(X)))
# net=MLP()
# net(X)
#2.sequential法    
class MySequential(nn.Module):
    def __init__(self,*args):
        super.__init__()
        for block in args:
            self.__modules[block]=block #成员字典
    def forward(self,X):
        for block in self._modules.values():
            X=block(X)
        return X
net=MySequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
net(X)
#或者
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args): #ennumerate会给迭代元素打上标记01 xx 02 xx
        # 这⾥，module是Module⼦类的⼀个实例。我们把它保存在'Module'类的成员
        # 变量_modules中。module的类型是OrderedDict
            self._modules[str(idx)] = module
        def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
            for block in self._modules.values():
                X = block(X)
            return X
#3.nn.Moudle和sequential 可以嵌套使用 在自定义的层里加self.xx=nn.sequential()即可


'''
1.任何一个层，一个神经网络都是Module的子类
module两个重要的函数：init forward
2.使用nn.Module方法比sequential更加灵活
3.net(x)相当于net.__call__(x),而__call__(x)调用了forward(x)(python语法糖)
4.enumerate 会给迭代的元素打上标记


'''
