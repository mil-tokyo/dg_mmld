from torch import nn
from torch.nn import functional as F
import torch

class MaximumSquareLoss(nn.Module):
    def __init__(self):
        super(MaximumSquareLoss, self).__init__()
    def forward(self, x):
        p = F.softmax(x, dim=1)
        b = (torch.mul(p, p))
        b = -1.0 *  b.sum(dim=1).mean() / 2
        return b
