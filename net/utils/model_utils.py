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


def normalize_adj_rc(adj: torch.Tensor) -> torch.Tensor:
    d_12 = torch.diag(torch.pow(torch.sum(adj, dim=1) + 1e-5, -1 / 2))
    norm_adj = d_12 @ adj @ d_12
    return norm_adj


def nonzero(adj: torch.Tensor) -> torch.Tensor:
    return (adj != 0).type(torch.float32)
