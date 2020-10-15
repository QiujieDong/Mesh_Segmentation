import numpy as np
import os
import torch

target_length = 9
input_arr = np.array([[2,2,4,2,2],[3,3,3,3,6],[8,4,4,4,4]])
# shp = input_arr.shape
# print(shp)

# npad1 = [(0, 0) for _ in range(len(shp))]
# print(npad1)
# npad = npad1
# npad[1] = (0, target_length - shp[1])
# print(npad)
# nnpad=np.pad(input_arr, pad_width=npad, mode='constant', constant_values=0)
# print(nnpad)
input_tensor = torch.from_numpy(input_arr)
print(input_tensor.data)
max_tensor = input_tensor.data.max(1)[1]
print(max_tensor.shape)
