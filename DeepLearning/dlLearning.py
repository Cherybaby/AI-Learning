import torch
import numpy as np

# 1D Tensor
t1 = torch.tensor([1, 2, 3])
print(t1)

# 2D Tensor
t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(t2)

# 3D Tensor
t3 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(t3)

# 从 NumPy 创建 Tensor
arr = np.array([1, 2, 3])
t_np = torch.tensor(arr)
print(t_np)