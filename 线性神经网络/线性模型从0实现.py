# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 09:28:03 2022

@author: Janus_yu
"""
import matplotlib
import random
import torch
from d2l import torch as d2l
print("生成数据集")
#

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
print("features:",features[0],'\nlable:',labels[0])
d2l.set_figsize()
d2l.plt.scatter(features[:,0].detach().numpy(),labels.detach().numpy(),1)

print("读取小批量")
i=0
#
def data_iter(batch_size,features,labels):
    numexamples=len(features)
    indices=list(range(numexamples))
    random.shuffle(indices)#把上行生成的下标全部打乱
    for i in range(0,numexamples,batch_size): #batchsize在这里是步长
        batch_indices=torch.tensor(
            indices[i:min(i+batch_size,numexamples)])
        yield features[batch_indices],labels[batch_indices]
batch_size=10
for x,y in data_iter(batch_size,features,labels):
   print(x,'\n',y)
   break



print('定义模型和参数') 
w=torch.normal(0, 0.01,size=(2,1),requires_grad=True) 
b=torch.zeros(1,requires_grad=True)
def linreg(x,w,b):#模型
    return torch.matmul(x,w)+b
def squared_loss(y_hat,y):#损失函数
    return (y_hat-y.reshape(y_hat.shape))**2/2
def sgd(params,lr,batch_size):
    with torch.no_grad():#优化算法：小批量随机梯度下降
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()
#训练过程
lr=0.03
num_epochs=3
net=linreg
loss=squared_loss   
for epoch in range(num_epochs):
    for x,y in data_iter(batch_size, features, labels):
        l=loss(net(x,w,b),y)
        l.sum().backward()
        sgd([w,b], lr, batch_size)
    with torch.no_grad():
        train_1=loss(net(features,w,b),labels)
        print(f'epoch {epoch + 1}, loss {float(train_1.mean()):f}')
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 

    
    