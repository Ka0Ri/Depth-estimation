import torch
import torch.nn.functional as F

a = torch.rand(3, 3, 2, 2)
print(a)
a = torch.transpose(a, 1, 2)
b = a.reshape(6, 6)
print(b)