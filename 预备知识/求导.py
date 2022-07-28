# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 16:00:55 2022

@author: Janus_yu
"""

import torch
x=torch.arange(4.0,requires_grad=True)
print(x)
'''
保存梯度
x=torch.arange(4.0)
x.requires_grad_(true)
等价于x=torch.arange(4.0,requires_grad=True)
'''
print('#计算y=2*X转置*X的梯度')
y=2*torch.dot(x,x)
y.backward()
print(x.grad)

print('#pytorch会积累梯度，清除之前的值')
#若不清零梯度会“累加”
x.grad.zero_()
y=x.sum()
y.backward()
print(x.grad)
print('#一般不对向量直接求导 sum()是标量')
x.grad.zero_()
y=x*x
y.sum().backward()
print(x.grad)
print('#将某些计算移动到记录的计算图之外')
x.grad.zero_()
y=x*x
u=y.detach()#分离梯度requires_grad=False
z=u*x
z.sum().backward()
print(x.grad==u) 

x.grad.zero_()
y.sum().backward()
print(x.grad==2*x)
#
print('构建函数需要python的控制流，我们仍然可以计算得到的变量')
def f(a):
    b=a*2
    while b.norm()<1000:
        b=b*2
    if b.sum()>0:
        c=b
    else:
        c=100*b
    return c
a=torch.randn(size=(),requires_grad=True)
d=f(a)
d.backward()
print(a.grad==d/a)
    























