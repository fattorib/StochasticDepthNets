import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from Stochastic_Depth_Nets.src.ResidualBlocks import StochasticDepthResNet
import time
import torch.autograd.profiler as profiler

from ptflops import get_model_complexity_info

# #------- Profiling
# # Use PyTorch tools to profile model for QA

# with torch.cuda.device(0):
#     model = StochasticDepthResNet(filters_list=[16, 32, 64], N=18, p_L=0.5)
#     model.eval()

#     macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
#                                              print_per_layer_stat=True, verbose=True)
#     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#     print('{:<30}  {:<8}'.format('Number of parameters: ', params))

x = [6, 6, 6, 6, 4]
a = np.array(np.unique(x, return_counts=True)).T


print(a)
print(a[0, 0])
