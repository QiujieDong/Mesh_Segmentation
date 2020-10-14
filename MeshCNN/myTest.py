import numpy as np
import os

target_length = 9
input_arr = np.array([[2,2,2,2,2],[3,3,3,3,3],[4,4,4,4,4]])
shp = input_arr.shape
print(shp)

npad1 = [(0, 0) for _ in range(len(shp))]
print(npad1)
npad = npad1
npad[1] = (0, target_length - shp[1])
print(npad)
nnpad=np.pad(input_arr, pad_width=npad, mode='constant', constant_values=0)
print(nnpad)

