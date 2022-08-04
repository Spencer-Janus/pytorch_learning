# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 20:03:31 2022

@author: Janus_yu
"""

#加载和保存张量
import torch 
from torch import nn
from torch.nn import functional as F

x=torch.arange(4)
torch.save(x,'x-file')

x2=torch.load('x-file')

print(x2)

#存一个list

y=torch.zeros(4)

torch.save([x,y],'x-files')
x2,y2=torch.load('x-files')
print((x2,y2))

#写入或读取从字符串映射到张量的字典

mydict={'x':x,'y':y}
torch.save(mydict,'mydict')
mydict2=torch.load('mydict')
print(mydict2)

#加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(20,256)
        self.output=nn.Linear(256,10)
    def forward(self,x):
        return self.output(F.relu(self.hidden(x)))
net=MLP()
x=torch.randn(size=(2,20))
y=net(x)

torch.save(net.state_dict(),'mlp.params')


#实例化原始模型的备份，直接读取参数
clone=MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()  #进入测试模式

Y_clone=clone(x)
Y_clone==y





