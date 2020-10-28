import torch
import torch.nn as nn

from .components import MLP


class Initializer(nn.Module):
    def __init__(self):
        super(Initializer, self).__init__()

    def forward(self, *input: Any, **kwargs: Any) -> T_co:
        pass


class ConfAwareMPNNKernel(nn.Module):
    def __init__(self):
        super(ConfAwareMPNNKernel, self).__init__()

    def forward(self, *input: Any, **kwargs: Any) -> T_co:
        pass


class InformedHamiltonianKernel(nn.Module):
    def __init__(self):
        super(InformedHamiltonianKernel, self).__init__()

    def forward(self, *input: Any, **kwargs: Any) -> T_co:
        pass


class ConfAwareReadout(nn.Module):
    def __init__(self):
        super(ConfAwareReadout, self).__init__()

    def forward(self, *input: Any, **kwargs: Any) -> T_co:
        pass


class ConformationComparison(nn.Module):
    def __init__(self):
        super(ConformationComparison, self).__init__()

    def forward(self, *input: Any, **kwargs: Any) -> T_co:
        pass
