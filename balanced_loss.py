import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torch
import torch.nn.functional as F




class W_CEL(nn.Module):
    def __init__(self, weight=None, Verbose=False):
        super(W_CEL, self).__init__()
        self.weight = weight 
        self.Verbose = Verbose
        
    def forward(self, input, target):
        class_sum = torch.sum(target, dim=0)
        positive_ratio = class_sum / target.size()[0]   # P / (P+N)
        negative_ratio = torch.ones(positive_ratio.size()).cuda() - positive_ratio # N / (P+N)
#         print(positive_ratio)
#         print(negative_ratio)

        loss = -(target * F.logsigmoid(input) * negative_ratio + (1 - target) * F.logsigmoid(-input) * positive_ratio)
        
        loss = loss.sum(dim=1) / input.size(1)
        ret = loss.mean()
        
        return ret