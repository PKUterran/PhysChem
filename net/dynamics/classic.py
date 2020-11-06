import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import Union, Tuple

from data.encode import get_default_atoms_massive_matrix
from net.utils.MaskMatrices import MaskMatrices


class KineticEnergy(nn.Module):
    def __init__(self, p_dim, h_dim=128):
        super(KineticEnergy, self).__init__()
        self.W = nn.Linear(p_dim, h_dim, bias=False)

    def forward(self, p, m):
        alpha = 1 / m
        pw = self.W(p)
        apwwp = alpha * (pw ** 2)
        if torch.isnan(apwwp.sum()):
            apwwp[torch.isnan(apwwp)] = 0
        t = torch.sum(apwwp, dim=1, keepdim=True)
        return t


class PotentialEnergy(nn.Module):
    def __init__(self, q_dim, h_dim=32, dropout=0.0, use_cuda=False):
        super(PotentialEnergy, self).__init__()
        self.use_cuda = use_cuda
        self.linear1 = nn.Linear(q_dim, h_dim, bias=True)
        self.softplus = nn.Softplus()

    def forward(self, m, q, vvm):
        norm_m = m
        mm = norm_m * norm_m.reshape([1, -1])
        eye = torch.eye(vvm.shape[1], dtype=torch.float32)
        if self.use_cuda:
            eye = eye.cuda()
        mask = vvm * mm
        delta_p = torch.unsqueeze(q, dim=0) - torch.unsqueeze(q, dim=1)
        root = self.linear1(delta_p)
        distance = (self.softplus(torch.sum(root ** 2, dim=2))) * (-eye + 1) + eye
        energy = mask * (distance ** -2 - distance ** -1)
        if torch.isnan(energy.sum()):
            energy[torch.isnan(energy)] = 0
        p = torch.sum(energy, dim=1, keepdim=True)
        return p


class DissipatedEnergy(nn.Module):
    def __init__(self, p_dim, h_dim=32):
        super(DissipatedEnergy, self).__init__()
        self.W = nn.Linear(p_dim, h_dim, bias=False)

    def forward(self, p, m):
        alpha2 = 1 / (m ** 2)
        pw = self.W(p)
        a2pwwp = alpha2 * (pw ** 2)
        if torch.isnan(a2pwwp.sum()):
            a2pwwp[torch.isnan(a2pwwp)] = 0
        f = torch.sum(a2pwwp, dim=1, keepdim=True)
        return f


class DissipativeHamiltonianDerivation(nn.Module):
    def __init__(self, p_dim: int, q_dim: int,
                 use_cuda=False, dropout=0.0):
        super(DissipativeHamiltonianDerivation, self).__init__()
        self.T = KineticEnergy(p_dim)
        self.U = PotentialEnergy(q_dim, dropout=dropout, use_cuda=use_cuda)
        self.F = DissipatedEnergy(p_dim)

    def forward(self, m: torch.Tensor, p: torch.Tensor, q: torch.Tensor, mask_matrices: MaskMatrices,
                return_energy=False, dissipate=True
                ) -> Union[Tuple[torch.Tensor, torch.Tensor],
                           Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        mvw = mask_matrices.mol_vertex_w
        vvm = mvw.t() @ mvw
        hamiltonians = self.T(p, m) + self.U(m, q, vvm)
        dissipations = self.F(p, m)
        hamilton = hamiltonians.sum()
        dissipated = dissipations.sum()
        dq = autograd.grad(hamilton, p, create_graph=True)[0]
        if dissipate:
            dp = -1 * (autograd.grad(hamilton, q, create_graph=True)[0] +
                       autograd.grad(dissipated, p, create_graph=True)[0] * m)
        else:
            dp = -1 * autograd.grad(hamilton, q, create_graph=True)[0]
        if return_energy:
            return dp, dq, hamiltonians, dissipations
        return dp, dq
