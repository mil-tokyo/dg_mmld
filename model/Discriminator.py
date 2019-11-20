import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd, reverse=True):
        ctx.lambd = lambd
        ctx.reverse=reverse
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.reverse:
            return (grad_output * -ctx.lambd), None, None
        else:
            return (grad_output * ctx.lambd), None, None

def grad_reverse(x, lambd=1.0, reverse=True):
    return GradReverse.apply(x, lambd, reverse)

class Discriminator(nn.Module):
    def __init__(self, dims, grl=True, reverse=True):
        if len(dims) != 4:
            raise ValueError("Discriminator input dims should be three dim!")
        super(Discriminator, self).__init__()
        self.grl = grl
        self.reverse = reverse
        self.model = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dims[2], dims[3]),
        )
        self.lambd = 0.0

    def set_lambd(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        if self.grl:
            x = grad_reverse(x, self.lambd, self.reverse)
        x = self.model(x)
        return x