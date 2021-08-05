import torch
import numpy as np

# a = torch.tensor([[1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1]])
# an=np.array([[1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1]])
b=torch.arange(10)
# bn=np.arange(10)
# cn=np.arange(10)
# c=torch.arange(10)
# # d=(a==c)
# # e=torch.log(torch.masked_select(b, d)/100)
# # print(torch.masked_select(b, a>5))
# # print(e)
# # print(torch.sum(e))

# # print((a == 1.).sum())
# print(torch.dot(b,c))
# print(np.dot(bn,cn))
# print(np.dot(an,bn))
# print(torch.tensordot(a,b,dims=([-1], [-1])))
# print(torch.inner(a,b))

# a = torch.arange(60.).reshape(3, 4, 5)
# b = torch.arange(24.).reshape(4, 3, 2)
# print(a)
# print(b)
# print(torch.tensordot(a, b, dims=([1, 0], [0, 1])))
b=torch.tensor([[1.,1.,1.],[1.,1.,1.]])
print(torch.logsumexp(b,[0,1]))