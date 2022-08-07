# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 15:48:27 2022

@author: Janus_yu
"""
'''
通过限制参数值的选择范围来控制模型容量
'''
import torch
from torch import nn
from d2l import torch as d2l

n_train,n_test,num_inputs,batch_size=20,100,200,5
true_w,true_b=torch.ones((num_inputs,1))*0.01,0.05
train_data=d2l.synthetic_data(true_w,true_b,n_train)
train_iter=d2l.load_array(train_data,batch_size)
test_data=d2l.synthetic_data(true_w, true_b,n_test)
test_iter=d2l.load_array(test_data, batch_size,is_train=False)

def init_params():
    w=torch.normal(0,1,size=(num_inputs,1),requires_grad=True)
    b=torch.zeros(1,requires_grad=True)
    return [w,b]
def l2_penalty(w):
    return torch.sum(w.pow(2))/2
def train(lambd):
    w,b=init_params()
    net,loss=lambda X:d2l.linreg(X,w,b),d2l.squared_loss 
    num_epochs,lr=100,0.003
    animator=d2l.Animator(xlabel='epochs',ylabel='loss',yscale='log',xlim=[5,num_epochs],legend=['train','test'])
    for epoch in range(num_epochs):
        for X,y in train_iter:
            l=loss(net(X),y)+lambd*l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w,b],lr,batch_size)
        if(epoch+1)%5==0:
            animator.add(epoch+1,(d2l.evaluate_loss(net, train_iter, loss),
                                  d2l.evaluate_loss(net, test_iter, loss)))
    
    print('w的L2范数是：', torch.norm(w).item())
train(lambd=0)                         
#简洁实现
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减，在下⾯的代码中，我们在实例化优化器时直接通过weight_decay指定weight decay超参数。默认情况下，
    #PyTorch同时衰减权重和偏移。这⾥我们只为权重设置了weight_decay，所以偏置参数b不会衰减。
    trainer = torch.optim.SGD([{"params":net[0].weight,'weight_decay': wd},{"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
    xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,(d2l.evaluate_loss(net, train_iter, loss),d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())
'''
    深度学习框架为了便于我们使⽤权重衰减，将权重衰减集成到优化算法中，以便与任何损失函数结合使⽤。此外，这种集成还有计算上的好处，允许在不增加任何额外的计算
    开销的情况下向算法中添加权重衰减。由于更新的权重衰减部分仅依赖于每个参数的当前值，因此优化器必须⾄少接触每个参数⼀次。
'''
