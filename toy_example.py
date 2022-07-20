import numpy as np
import torch

'''
a = np.array([1, 2, 3, 5, 3,6])
b = np.empty(20)
print(b)
c = np.empty((20, ))
print(c)
d = np.empty((20, )+ (3, ), dtype=np.int32)
print(d)

aa = torch.tensor([[2],
                   [-1]])
print(aa)
print(aa.shape)
bb = aa > 0
print(bb)
cc = bb.expand(2, 4)
print(cc)
'''

'''
a = torch.tensor([[2,4,4,3],
                  [3,-2,1,6],
                 [4,1,2,3]])
print(a.shape)
b = torch.tensor([[True, True, True, True],
                 [False, False, False, False],
                  [True, True, True, True]])
print(b.shape)

c = a[b]
print(c)
print(c.shape)
'''

a = 2 ** 4
print(a)