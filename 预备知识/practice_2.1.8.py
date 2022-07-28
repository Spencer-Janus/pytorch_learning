import torch
x=torch.arange(12,dtype=torch.float32).reshape(3,4)
y=torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
print(x)
print(y)
print(x<y)
print(x>y)
print(x+y)

z=torch.zeros((3,4))
z[:]=y
z=z.reshape(1,2,6)
print(z)
y=y.reshape(2,1,6)
print(y)
print(z+y)
'''
结果预期相同，但要注意：
The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 2
非单一元素维度
他们最大的维度数必须相同
'''