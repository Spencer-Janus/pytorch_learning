import torch
x=torch.arange(12)
print(x)
print(type(x))#张量tensor-数组
print(x.shape) #张量的形状和元素的总数
print(x.numel())
X=x.reshape(3,4)#改变张量的形状
print(X)
print(X.shape)
print(torch.zeros(2,3,4))
print(torch.randn(3, 4))
y=torch.tensor([[1,2,3,4],[5,6,7,8],[4,3,2,1]])
print(y)
print(y.shape)
x=torch.tensor([1.0,2,4,8])
y=torch.tensor([1,2,3,4])#几个方括号就是几维张量
print(x+y)
#张量连结
x=torch.arange(12,dtype=torch.float32).reshape(3,4)
y=torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
print(torch.cat((x,y),dim=0))#按y方向拼接
print(torch.cat((x,y),dim=1))#按x方向拼接
'''
x=torch.arange(12,dtype=torch.float32).reshape(1,3,4)
y=torch.tensor([[[2.0,1,4,3],[1,2,3,4],[4,3,2,1]]])
print(torch.cat((x,y),dim=0))#按y方向拼接
print(torch.cat((x,y),dim=1))#按x方向拼接
print(torch.cat((x,y),dim=2))
'''
z=(x==y)#逻辑运算符构建二元张量
print(z)
X=(x.sum())#求和产生一个元素的张量
print(X)
print(X.shape)

#广播机制
a=torch.arange(3).reshape(3,1)
b=torch.arange(2).reshape(1,2)
c=a+b
print(c)#1->2 1->3 复制成一样形状的矩阵

#元素访问
print(c[-1])#最后一行
print(c[1:3:1])#1到2行 begin end step(默认为1)
print(c[0,0])#0行0列的元素
print(c[0:2,0:1])
c[0:2,0:1]=12 #左边选0到二行，右边选0列
print(c)

#一些操作会导致为新结果分配内存
before=id(a)
a=a+b
print(id(a)==before)
#执行原地操作
z=torch.zeros_like(a)
print("ida",id(a))
a[:]=a+b
print("ida",id(a))
'''
如果在后续计算中没有重复使用X，我们也可以使用X[:] = X + Y或 X += Y来减少操作的内存开销。
'''

#转换为numpy张量
A=a.numpy()
b=torch.tensor(A)#在转换回来
#将大小为1的张量转换为python标量
a=torch.tensor([3.5])
A=float(a)
B=int(a)


import os


data_file='house_tiny.csv'
with open('house_tiny.csv', 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')


import pandas as pd
data=pd.read_csv(data_file)
print(data)

inputs,outputs=data.iloc[:,0:2],data.iloc[:,2]
print(inputs)
print(outputs)
inputs=inputs.fillna(inputs.mean())#NA换成均值
print(inputs)
inputs=pd.get_dummies(inputs,dummy_na=True)
print(inputs)

x,y=torch.tensor(inputs.values),torch.tensor(outputs.values)
'''
广播机制的条件:
两个张量进行广播机制的条件
1 两个张量都至少有一个维度
2 按从右往左顺序看两个张量的每一个维度，x和y每个对应着的两个维度都需要能够匹配上。什么情况下算是匹配上了？满足下面的条件就可以：
      a.这两个维度的大小相等
      b. 某个维度 一个张量有，一个张量没有
      c.某个维度 一个张量有，一个张量也有但大小是1
x=torch.empty(5,3,4,1)
y=torch.empty( 3,1,1)
如上面代码中，首先将两个张量维度向右靠齐，从右往左看，两个张量第四维大小相等，都为1，满足上面条件a;第三个维度大小不相等，但第二个张量第三维大小为1，
满足上面条件b;第二个维度大小相等都为3，满足上面条件a;第一个维度第一个张量有，第二个张量没有，满足上面条件b，因此两个张量每个维度都符合上面广播条件，因此可以进行广播。
两个张量维度从右往左看，如果出现两个张量在某个维度位置上面，维度大小不相等，且两个维度大小没有一个是1，那么这两个张量一定不能进行广播。
'''














