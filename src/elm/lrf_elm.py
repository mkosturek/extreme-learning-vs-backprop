import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair, _quadruple
import torch.nn.functional as F


class LocalReceptiveField(nn.Module):

    def __init__(self, in_channels, out_channels,
                 conv_kernel_size=3, conv_stride=1,
                 pool_kernel_size=3, pool_stride=2):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              conv_kernel_size, conv_stride)
        self.pool = SquareRootPooling(pool_kernel_size, pool_stride)

    def forward(self, x):
        # parts = []
        # i = 0
        # trial = self.conv(x[:1])
        # parts = torch.zeros(x.shape[0], trial.shape[1],
        #                     trial.shape[2], trial.shape[3])
        # while i < len(x):
        #     parts[i:i+10000] = self.conv(x[i:i+10000])
        #     i += 10000
        #
        # return self.pool(parts)
        return self.pool(self.conv(x))

    def orthogonalise_kernels(self):
        normal = torch.distributions.Normal(0, 0.01)
        kernels = normal.sample((self.conv.out_channels,
                                 (self.conv.in_channels
                                  * self.conv.kernel_size[0]
                                  * self.conv.kernel_size[1]))
                                )

        if kernels.shape[1] > kernels.shape[0]:
            _, _, kernels = torch.svd(kernels)
            kernels = kernels.t()
        else:
            kernels, _, _ = torch.svd(kernels)
        kernels = kernels.reshape(self.conv.out_channels,
                                  self.conv.in_channels,
                                  self.conv.kernel_size[0],
                                  self.conv.kernel_size[1])
        self.conv.weight = nn.Parameter(kernels)


class SquareRootPooling(nn.Module):
    """ Sqrt pool module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(SquareRootPooling, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).pow(2).sum(dim=-1).sqrt()
        return x
