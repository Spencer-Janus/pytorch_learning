# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:31:49 2022

@author: Janus_yu
"""
import torch
#1
A=torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(A.T.T)
#2
B=torch.tensor([[1,1,1],[1,0,1],[1,1,1]])
print(((A+B).T)==((A.T)+(B.T)))
#3
print(A+(A.T))
print(B+(B.T))
'''
总是对称的 A+A转置 整体的转置=A转置+A 
矩阵加法具有交换律
'''
#4
#len(A)为A第一个维度上的大小
#5总是对应第一个轴
#6
print(A.sum(axis=1))
print(A/A.sum(axis=1))
#7
B=torch.arange(24,dtype=torch.float).reshape(2,3,4)
print(B)
print(B.sum(axis=0))
print(B.sum(axis=1))
print(B.sum(axis=2))
#8
C=torch.ones(2,2,2)
print(torch.norm(C))
print(C.ndim)
print(C.shape)

print(C.dtype)