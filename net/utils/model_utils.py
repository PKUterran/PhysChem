import torch
import torch.nn as nn
from typing import Callable


def activation_select(act: str) -> Callable:
    if act == 'no':
        activate = lambda x: x
    elif act == 'sigmoid':
        activate = nn.Sigmoid()
    elif act == 'tanh':
        activate = nn.Tanh()
    elif act == 'softmax':
        activate = nn.Softmax(dim=-1)
    else:
        assert False, 'Undefined activation {}'.format(act)

    return activate


def normalize_adj_r(adj: torch.Tensor) -> torch.Tensor:
    d_1 = torch.diag(torch.pow(torch.sum(adj, dim=1) + 1e-5, -1))
    norm_adj = d_1 @ adj
    return norm_adj
