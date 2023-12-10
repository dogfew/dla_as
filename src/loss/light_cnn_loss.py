import torch
import torch.nn as nn


class AngularSoftmax(nn.Module):
    def __init__(self):
        super(AngularSoftmax, self).__init__()
        self.loss = ...

    def forward(self, x, y):
        return self.loss(x, y)
