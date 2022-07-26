# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20  16:54:52 2022

@author: Janus_yu
"""
import torch
import mnist
from IPython import display 
from d2l import torch as d2l
batchsize=256
train_iter,test_iter=mnist.load_data_fashion_mnist(batchsize)
num_inputs=784 #28*28 将图象拉长
num_outputs=10
#
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
#
w=torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b=torch.zeros(num_outputs,requires_grad=True)
def softmax(X):
    X_exp=torch.exp(X) #对矩阵每个元素求exp
    partition=X_exp.sum(1,keepdim=True)
    return X_exp/partition 
#实现softmax回归模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1,w.shape[0])),w)+b)

#实现交叉熵损失函数
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])
print(cross_entropy(y_hat, y))
    
#y_hat[[],[]],里面第一个[]代表行号，第二个[]代表列号

def accuracy(y_hat,y):#计算预测正确的数量
    if len(y_hat.shape)>1 and y_hat .shape[1]>1:
        y_hat=y_hat.argmax(axis=1)
    cmp=y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())
print(accuracy(y_hat, y) / len(y))
def evaluate_accuracy(net,data_iter):
#计算在指定数据集上模型的精度
    if isinstance(net, torch.nn.Module):
        net.eval() #将模型设置为评估模式
    metric=Accumulator(2)#正确预测数，预测总数
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]


class Accumulator:

    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[a+float(b)for a,b in zip(self.data,args)]
    def reset(self):
        self.data=[0.0]*len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]

print(evaluate_accuracy(net,test_iter))


def train_epoch_ch3(net,train_iter,loss,updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric=Accumulator(3)
    for X,y in train_iter:
        y_hat=net(X)
        l=loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()  #更新参数
            metric.add(
                float(l)*len(y),accuracy(y_hat,y),y.size().numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()),accuracy(y_hat, y),y.numel())
    return metric[0]/metric[2],metric[1]/metric[2]
class Animator: #在动画中绘制数据
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,figsize=(3.5, 2.5)):
        if legend is None:
           legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:  
            self.axes = [self.axes, ]
# 使⽤lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(self.axes[0], xlabel, ylabel, xlim, 
                    ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
    def add(self, x, y):
# 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
              x = [x] * n
        if not self.X:
              self.X = [[] for _ in range(n)]
        if not self.Y:
              self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
lr = 0.1
def updater(batch_size):
    return d2l.sgd([w, b], lr, batch_size)
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
    train_loss, train_acc = train_metrics
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
'''
3.2 Pytorch实现nn.CrossEntropyLoss
实际使用中需要注意几点:

torch.nn.CrossEntropyLoss(input, target)中的标签target使用的不是one-hot形式，而是类别的序号。形如 target = [1, 3, 2] 表示3个样本分别属于第1类、第3类、第2类。（单标签多分类问题）
torch.nn.CrossEntropyLoss(input, target)的input是没有归一化的每个类的得分，而不是softmax之后的分布。
'''
