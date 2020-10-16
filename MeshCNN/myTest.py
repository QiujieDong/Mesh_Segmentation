import numpy as np
import os
import torch
from tempfile import mkstemp
from shutil import move

file = "/home/qiujie/meshSegmentation/MeshCNN/test.py"
# target_length = 9
# input_arr = np.array([[2,2,4,2,2],[3,3,3,3,6],[8,4,4,4,4]])
# # shp = input_arr.shape
# # print(shp)

# # npad1 = [(0, 0) for _ in range(len(shp))]
# # print(npad1)
# # npad = npad1
# # npad[1] = (0, target_length - shp[1])
# # print(npad)
# # nnpad=np.pad(input_arr, pad_width=npad, mode='constant', constant_values=0)
# # print(nnpad)
# input_tensor = torch.from_numpy(input_arr)
# print(input_tensor.data)
# max_tensor = input_tensor.data.max(1)[1]
# print(max_tensor.shape)

# with open(file) as old_file:#打开file文件
#     for line in old_file: 
#         print(line)

file = "/home/qiujie/meshSegmentation/MeshCNN/1.py"
new_file = "/home/qiujie/meshSegmentation/MeshCNN/data/2.py"
os.remove(file)
move(new_file, file)    
print("1")