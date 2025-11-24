import torch

x = torch.arange(12,dtype=torch.float32)
# print(x)
# print(x.numel())
# print(x.shape)
X = x.reshape(3,4)
# print(X.shape)
# print(torch.zeros((2,3,4)))
# print(X)
X[:2, :3] = 0
# print(X)
X1 = torch.arange(12,dtype=torch.float32).reshape(3,4)
X2 = torch.tensor([[1.1,2.2,3.3,4.4],[5.5,6.6,7.7,8.8],[9.9,10.0,11.1,12.2]])
# print(torch.cat((X1,X2),dim=0))
# print(torch.cat((X1,X2),dim=1))

#broadcasting
a = torch.arange(3).reshape((3,1))
b = torch.arange(2).reshape((1,2))
print(a)
print(b)
print(a+b)

before = id(X1)
# change addition to in-place addition
X1 = X1 + X2
# not change addition to in-place addition
# X1 += X2
# after = id(X1)
# print(X1)
# print(before == after)  # True

X = torch.arange(4.0)
X.requires_grad_(True)
print(X.grad)
y = 2 * torch.dot(X, X)
print(y)
y.backward()
print(X.grad)