# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 16:08:02 2022

@author: Janus_yu
"""
from torch import nn
import torch

#张量与GPU
X = torch.ones(2, 3, device=torch.device('cuda'))

print(X)

#网络与GPU
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=torch.device('cuda'))