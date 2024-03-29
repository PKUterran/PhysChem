import torch
import torch.nn as nn
from typing import Tuple

from net.utils.MaskMatrices import MaskMatrices


class Force(nn.Module):
    ESP = 1e-6

    def __init__(self, v_dim: int, e_dim: int, q_dim: int, h_dim=32,
                 use_cuda=False, dropout=0.0):
        super(Force, self).__init__()
        self.fb_linear1 = nn.Linear(v_dim + e_dim, h_dim, bias=False)
        self.fb_relu = nn.LeakyReLU()
        self.fb_linear2 = nn.Linear(h_dim, 1)
        self.fb_tanh = nn.Tanh()

        self.fr_linear = nn.Linear(q_dim, h_dim)
        self.fr_relu = nn.Softplus()
        self.fr_tanh = nn.Tanh()

    def forward(self, v: torch.Tensor, e: torch.Tensor, m: torch.Tensor, q: torch.Tensor,
                mask_matrices: MaskMatrices) -> torch.Tensor:
        # bond force
        vew1 = mask_matrices.vertex_edge_w1  # shape [n_vertex, n_edge]
        vew2 = mask_matrices.vertex_edge_w2  # shape [n_vertex, n_edge]
        vew_u = torch.cat([vew1, vew2], dim=1)  # shape [n_vertex, 2 * n_edge]
        vew_v = torch.cat([vew2, vew1], dim=1)  # shape [n_vertex, 2 * n_edge]
        e2 = torch.cat([e, e])  # shape [2 * n_edge, e_dim]
        uv_e = torch.cat([(vew_u + vew_v).t() @ v, e2], dim=1)
        delta_q = vew_u.t() @ q - vew_v.t() @ q
        unit_f_bond = delta_q / (torch.norm(delta_q, dim=1, keepdim=True) + self.ESP)
        value_f_bond = self.fb_tanh(self.fb_linear2(self.fb_relu(self.fb_linear1(uv_e))))
        f_bond = vew_u @ (unit_f_bond * value_f_bond)

        # relative force
        mvw = mask_matrices.mol_vertex_w
        vvm = mvw.t() @ mvw
        mm = m * m.reshape([1, -1])
        vv_massive_mask = vvm * mm
        delta_q = torch.unsqueeze(q, dim=1) - torch.unsqueeze(q, dim=0)
        delta_d = self.fr_relu(self.fr_linear(delta_q)).norm(dim=2)
        unit_f_rela = delta_q / (torch.norm(delta_q, dim=2, keepdim=True) + self.ESP)
        value_f_rela = self.fr_tanh((delta_d ** -2 - delta_d ** -1) * vv_massive_mask).unsqueeze(2)
        f_rela = (unit_f_rela * value_f_rela).sum(dim=1)

        f = f_bond + f_rela
        single_mask = vew_u.sum(dim=1) == 0
        f[single_mask] = f[single_mask].detach()
        return f


class NewtonianDerivation(nn.Module):
    def __init__(self, v_dim: int, e_dim: int, p_dim: int, q_dim: int,
                 use_cuda=False, dropout=0.0):
        super(NewtonianDerivation, self).__init__()
        assert p_dim == q_dim
        self.force = Force(v_dim, e_dim, q_dim, use_cuda=use_cuda, dropout=dropout)

    def forward(self, v: torch.Tensor, e: torch.Tensor, m: torch.Tensor, p: torch.Tensor, q: torch.Tensor,
                mask_matrices: MaskMatrices
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        v = torch.sigmoid(v)
        e = torch.sigmoid(e)
        # dq / dt = v = p / m
        dq = p / m

        # dp / dt = F
        dp = self.force(v, e, m, q, mask_matrices)

        return dp, dq
