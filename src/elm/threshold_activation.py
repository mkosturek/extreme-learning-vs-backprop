import torch
import torch.nn


class Threshold(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x > 0).to(torch.float32)
