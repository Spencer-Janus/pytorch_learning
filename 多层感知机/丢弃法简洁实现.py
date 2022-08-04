# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 20:13:33 2022

@author: Janus_yu
"""
import torch
from torch import nn
from d2l import torch as d2l
dropout1, dropout2 = 0.2, 0.5
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
net = nn.Sequential(nn.Flatten(),nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout1),
                    # 在第⼀个全连接层之后添加⼀个dropout层
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    # 在第⼆个全连接层之后添加⼀个dropout层
                    nn.Dropout(dropout2),
                    nn.Linear(256, 10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights);
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
