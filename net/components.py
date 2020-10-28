import torch
import torch.nn as nn
from typing import Tuple

from .utils.MaskMatrices import MaskMatrices


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: list = None, activation: str = 'no',
                 use_cuda=False, bias=True, residual=False):
        super(MLP, self).__init__()
        self.use_cuda = use_cuda
        self.residual = residual

        if not hidden_dims:
            hidden_dims = []
        in_dims = [in_dim] + hidden_dims
        out_dims = hidden_dims + [out_dim]
        self.linears = nn.ModuleList([nn.Linear(i, o, bias=bias) for i, o in zip(in_dims, out_dims)])
        if activation == 'no':
            self.activate = lambda x: x
        elif activation == 'sigmoid':
            self.activate = nn.Sigmoid()
        elif activation == 'tanh':
            self.activate = nn.Tanh()
        elif activation == 'softmax':
            self.activate = nn.Softmax(dim=-1)
        else:
            assert False, 'Undefined activation {} in net.layers.MLP'.format(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear in self.linears:
            x2 = linear(x)
            if self.residual:
                x = torch.cat([x, x2])
            else:
                x = x2
        x = self.activate(x)
        return x


class NaiveDynMessage(nn.Module):
    def __init__(self, hv_dim: int, he_dim: int, mv_dim: int, me_dim: int, p_dim: int, q_dim: int,
                 use_cuda=False):
        super(NaiveDynMessage, self).__init__()
        self.use_cuda = use_cuda

        self.neighbor = nn.Linear(p_dim + q_dim + he_dim + hv_dim, mv_dim)
        self.n_relu = nn.LeakyReLU()
        self.a_relu = nn.ELU()
        self.link = nn.Linear(hv_dim + p_dim + q_dim + hv_dim, me_dim)
        self.l_relu = nn.LeakyReLU()

    def forward(self, hv_ftr: torch.Tensor, he_ftr: torch.Tensor, p_ftr: torch.Tensor, q_ftr: torch.Tensor,
                mask_matrices: MaskMatrices
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        vew1 = mask_matrices.vertex_edge_w1
        vew2 = mask_matrices.vertex_edge_w2

        rv1_ftr = vew1.t() @ hv_ftr
        rv2_ftr = vew2.t() @ hv_ftr
        rp1_ftr = vew1.t() @ p_ftr
        rp2_ftr = vew2.t() @ p_ftr
        rq1_ftr = vew1.t() @ q_ftr
        rq2_ftr = vew2.t() @ q_ftr

        n1_ftr = self.neighbor(torch.cat())
        mv_ftr = None

        me_ftr = self.link(torch.cat([rv1_ftr, rp1_ftr - rp2_ftr, rq1_ftr - rq2_ftr, rv2_ftr]))
        me_ftr = self.l_relu(me_ftr)

        return mv_ftr, me_ftr


class NaiveUnion(nn.Module):
    def __init__(self, h_dim: int, m_dim: int,
                 use_cuda=False, bias=True):
        super(NaiveUnion, self).__init__()
        self.linear = nn.Linear(h_dim + m_dim, h_dim, bias=bias)
        self.activate = nn.LeakyReLU()

    def forward(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        h = self.linear(torch.cat([h, m]))
        h = self.activate(h)
        return h











