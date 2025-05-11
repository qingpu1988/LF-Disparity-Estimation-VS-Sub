import functools

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


class _routing(nn.Module):

    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        x = torch.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x)


class CondConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CondConv2D, self).__init__()
        num_experts = 4
        dropout_rate = 0.2
        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self._routing_fn = _routing(in_channels, num_experts, dropout_rate)
        kernel_size = _pair(kernel_size)
        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels, *kernel_size))

    def forward(self, inputs, stride, dilation, padding):
        b, _, _, _ = inputs.size()
        res = []
        for input in inputs:
            input = input.unsqueeze(0)
            pooled_inputs = self._avg_pooling(input)
            routing_weights = self._routing_fn(pooled_inputs)
            kernels = torch.sum(routing_weights[:, None, None, None, None] * self.weight, 0)
            out = F.conv2d(input, weight=kernels, stride=stride, dilation=dilation, padding=padding)
            res.append(out)
        return torch.cat(res, dim=0)
