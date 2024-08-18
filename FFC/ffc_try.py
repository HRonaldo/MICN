import math
from ffc import FourierUnit,FourierUnit_1d,SpectralTransform,SpectralTransform_1d,FFC,FFC_1d,FFC_BN_ACT,FFC_BN_ACT_1d
import torch

# 定义每个维度的大小
batch_size = 10
channels = 128
height = 32
width = 32

# 使用torch.randn来创建一个四维的张量，其元素值是从标准正态分布中随机采样得到的
four_dim_tensor = torch.randn(batch_size, channels, height, width)

# 打印张量的大小
print(four_dim_tensor.size())

ratio_gin=0.5
ratio_gout=0.5
f1 = FFC_BN_ACT(channels,channels,kernel_size=1,ratio_gin=ratio_gin,ratio_gout=ratio_gout)

in_cg = int(channels * ratio_gin)
in_cl = channels - in_cg

x = f1(torch.split(four_dim_tensor,[in_cl,in_cg],dim=1))

print(torch.cat(x,dim=1).shape)

# # 定义每个维度的大小
batch_size = 10
channels = 128
seq_len = 32

# 使用torch.randn来创建一个四维的张量，其元素值是从标准正态分布中随机采样得到的
tensor1 = torch.randn(batch_size, channels, seq_len)

# 打印张量的大小
print(tensor1.size())


ratio_gin=0.5
ratio_gout=0.5
f2 = FFC_BN_ACT_1d(channels,channels,kernel_size=1,ratio_gin=ratio_gin,ratio_gout=ratio_gout)

in_cg = int(channels * ratio_gin)
in_cl = channels - in_cg

y = f2(torch.split(tensor1,[in_cl,in_cg],dim=1))

print(torch.cat(y,dim=1).shape)