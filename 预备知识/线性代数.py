import torch
A=torch.arange(20).reshape(5,4)
print(A.T)#不改变A本身


X=torch.arange(24).reshape(2,3,4)
print(X)
A=torch.arange(20,dtype=(torch.float32)).reshape(5,4)
B=A.clone()
print(A)
'''
通过分配新内存，将A的副本给B。此时改变B不会对A影响
'''
print(A*B)#哈达玛积 两个矩阵对应元素相乘
a=2
print(a*A)#标量和矩阵运算
print(a+A)

x=torch.arange(4,dtype=torch.float32)
print(x.sum())#不管x的形状，sum永远是标量
#2.3.6降维
A=torch.arange(20*2,dtype=(torch.float32)).reshape(2,5,4)
print(A)
print(A.sum())
print("按一个维度求和：")
A_sumaxis0=A.sum(axis=0)#按一个维度进行求和
A_sumaxis1=A.sum(axis=1)
A_sumaxis2=A.sum(axis=2)
print(A_sumaxis0)#5*4
print(A_sumaxis1)#2*4
print(A_sumaxis2)#2*5
print("按两个维度求和")
print(A.sum([0,1]))#按两个维度求和
print(A.mean())#求均值，值为浮点数
print(A.sum()/A.numel())
print(A.mean(axis=0))
print(A.mean(axis=1))
#非降维求和
print("非降维求和")
sum_A_0=A.sum(axis=0,keepdims=True)#不丢失维度
sum_A_1=A.sum(axis=1,keepdims=True)
print(sum_A_0)#1*5*4
print(sum_A_1)#2*1*4
print(A/sum_A_0)#广播机制：两个矩阵维度必须一样
print(A.cumsum(axis=0))#非降维 累加求和
print(A.cumsum(axis=1))
print(A.cumsum(axis=2))
#2.3.7点积
#对一维向量做点积  dot(x,y)
print("一维点积")
x=torch.tensor([1.,2.,3.,4.])
y=torch.ones(4,dtype=(torch.float))
print(torch.dot(x,y))

#2.3.8矩阵和向量乘法mv(A,x)
print("矩阵和向量的乘法")
A=torch.ones(16,dtype=(torch.float)).reshape(4,4)
print(torch.mv(A,x))
#2.3.9矩阵乘法
print("矩阵乘法")
print(torch.mm(A,A))
#2.3.10范数：向量或矩阵的长度
print("向量的L2范数:元素平方和的平方根")
x=torch.tensor([1.,2.,3.,4.])
print(torch.norm(x))
print("向量的L1范数:元素绝对值的和两种方法：")
print(torch.abs(x).sum())
print(torch.norm(x,1))
print("矩阵的F范数")
B=torch.tensor([[1.,2.,3.],[4.,5.,6.]])
print(torch.norm(torch.ones(4,4)))
print(torch.norm(B))






