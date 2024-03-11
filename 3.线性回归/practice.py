import random
import torch
from d2l import torch as d2l

# 1.生成合成数据集
def synthetic_data(w,b,num_examples):
    x = torch.normal(0,1,())