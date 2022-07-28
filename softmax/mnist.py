# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:58:02 2022

@author: Janus_yu
"""

import matplotlib as plt
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
d2l.use_svg_display() #用svg显示图片 清晰度高

#通过ToTensor实例将图片转换为32位浮点数格式 #transforms.ToTensor()的操作对象有PIL格式的图像以及numpy（即cv2读取的图像也可以）这两种
#并除以255使得所有像素的值均在0到1之间

trans=transforms.ToTensor()
mnist_train=torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=True)
mnist_test=torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=True)
print("num_train",len(mnist_train))
print("num_test",len(mnist_test))

#mnist_[0][0]里存储的是图片 .shape为1*28*28 通道数：1 黑白图片 28x28  [][1]存储的是图片类型0-9分别代表 10种
#mnist 59999*2 
def get_fashion_mnist_labels(labels): 
#返回Fashion-MNIST数据集的文本标签
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
#绘制图像列表
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
# 图⽚张量
            ax.imshow(img.numpy())
        else:
    # PIL图⽚
            ax.imshow(img)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));

batch_size = 256
def get_dataloader_workers(): 
    return 0
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
'''
timer = d2l.Timer()
if __name__ == "__main__":
    for X, y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec')
'''
def load_data_fashion_mnist(batch_size,resize=None):
    trans=[transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans=transforms.Compose(trans)
    mnist_train=torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=True)
    mnist_test=torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,num_workers=get_dataloader_workers()))
