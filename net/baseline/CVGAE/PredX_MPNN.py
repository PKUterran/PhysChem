import torch
import torch.nn as nn
from typing import Tuple, Union

from net.components import MLP, GCN
from net.utils.MaskMatrices import MaskMatrices
from net.utils.model_utils import normalize_adj_rc


class NaiveMPNN(nn.Module):
    def __init__(self, hv_dim, he_dim, out_dim, use_cuda=False):
        super(NaiveMPNN, self).__init__()
        self.edge_attend = nn.Linear(hv_dim + he_dim + hv_dim, 1)
        self.edge_act = nn.Sigmoid()
        self.gcn = GCN(hv_dim, hv_dim, activation='tanh', use_cuda=use_cuda)
        self.remap = MLP(hv_dim + hv_dim, out_dim, hidden_dims=[hv_dim], use_cuda=use_cuda, dropout=0.2)

    def forward(self, hv_ftr: torch.Tensor, he_ftr: torch.Tensor,
                mask_matrices: MaskMatrices) -> torch.Tensor:
        vew1 = mask_matrices.vertex_edge_w1  # shape [n_vertex, n_edge]
        vew2 = mask_matrices.vertex_edge_w2  # shape [n_vertex, n_edge]
        vew_u = torch.cat([vew1, vew2], dim=1)  # shape [n_vertex, 2 * n_edge]
        vew_v = torch.cat([vew2, vew1], dim=1)  # shape [n_vertex, 2 * n_edge]
        hv_u_ftr = vew_u.t() @ hv_ftr  # shape [2 * n_edge, hv_dim]
        hv_v_ftr = vew_v.t() @ hv_ftr  # shape [2 * n_edge, hv_dim]
        he2_ftr = torch.cat([he_ftr, he_ftr])  # shape [2 * n_edge, he_dim]
        uev_ftr = torch.cat([hv_u_ftr, he2_ftr, hv_v_ftr], dim=1)  # shape [2 * n_edge, hv_dim + he_dim + hv_dim]

        edge_weight = self.edge_act(self.edge_attend(uev_ftr))  # shape [2 * n_edge, 1]
        adj = vew_u @ (vew_v * edge_weight.view(-1)).t()  # shape [n_vertex, n_vertex]
        adj = normalize_adj_rc(adj)
        hidden = self.gcn.forward(hv_ftr, adj)  # shape [n_vertex, hv_dim]
        out = self.remap(torch.cat([hv_ftr, hidden], dim=1))  # shape [n_vertex, out_dim]
        return out


class CVGAE(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int, config: dict,
                 use_cuda=False):
        super(CVGAE, self).__init__()
        self.use_cuda = use_cuda
        hv_dim = config['HV_DIM']
        he_dim = config['HE_DIM']
        self.hv_dim = hv_dim

        self.v_linear = nn.Linear(atom_dim, hv_dim, bias=True)
        self.vp_linear = nn.Linear(atom_dim + 3, hv_dim, bias=True)
        self.v_act = nn.Tanh()
        self.e_linear = nn.Linear(bond_dim, he_dim, bias=True)
        self.e_act = nn.Tanh()
        self.prior_mpnn = NaiveMPNN(hv_dim, he_dim, 2 * hv_dim, use_cuda=use_cuda)
        self.post_mpnn = NaiveMPNN(hv_dim, he_dim, 2 * hv_dim, use_cuda=use_cuda)
        self.pred_mpnn = NaiveMPNN(hv_dim, he_dim, 2 * hv_dim, use_cuda=use_cuda)

    def forward(self, atom_ftr: torch.Tensor, bond_ftr: torch.Tensor,
                mask_matrices: MaskMatrices, is_training=True, given_pos: torch.Tensor = None
                ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        assert not is_training or given_pos is not None
        hv_ftr = self.v_act(self.v_linear(atom_ftr))
        he_ftr = self.e_act(self.e_linear(bond_ftr))

        prior_z_out = self.prior_mpnn.forward(hv_ftr, he_ftr, mask_matrices)
        prior_z_mu, prior_z_lsgms = torch.split(prior_z_out, [self.hv_dim, self.hv_dim], dim=1)
        prior_z_sample = self._draw_sample(prior_z_mu, prior_z_lsgms)

        if is_training or given_pos is not None:
            hvp_ftr = self.v_act(self.vp_linear(torch.cat([atom_ftr, given_pos], dim=1)))
            post_z_out = self.post_mpnn.forward(hvp_ftr, he_ftr, mask_matrices)
            post_z_mu, post_z_lsgms = torch.split(post_z_out, [self.hv_dim, self.hv_dim], dim=1)
            post_z_sample = self._draw_sample(post_z_mu, post_z_lsgms)
            post_x_out = self.pred_mpnn.forward(hv_ftr + post_z_sample, he_ftr, mask_matrices)
            if not is_training:  # evaluating with UFF
                return post_x_out
            # training with ground truth
            klds_z = self._kld(post_z_mu, post_z_lsgms, prior_z_mu, prior_z_lsgms)
            klds_0 = self._kld_zero(prior_z_mu, prior_z_lsgms)
            return post_x_out, klds_z, klds_0
        else:  # evaluating without UFF
            prior_x_out = self.pred_mpnn.forward(hv_ftr + prior_z_sample, he_ftr, mask_matrices)
            return prior_x_out

    def _draw_sample(self, mu: torch.Tensor, lsgms: torch.Tensor, T=1):
        epsilon = torch.normal(torch.zeros(size=lsgms.shape, dtype=torch.float32), 1.)
        if self.use_cuda:
            epsilon = epsilon.cuda()
        sample = torch.mul(torch.exp(0.5 * lsgms) * T, epsilon)
        sample = torch.add(mu, sample)
        return sample

    @staticmethod
    def _kld(mu0, lsgm0, mu1, lsgm1):
        var0 = torch.exp(lsgm0)
        var1 = torch.exp(lsgm1)
        a = torch.div(var0 + 1e-5, var1 + 1e-5)
        b = torch.div(torch.pow(mu1 - mu0, 2), var1 + 1e-5)
        c = torch.log(torch.div(var1 + 1e-5, var0 + 1e-5) + 1e-5)
        kld = 0.5 * torch.sum(a + b - 1 + c, dim=1)
        return kld

    @staticmethod
    def _kld_zero(mu, lsgm):
        a = torch.exp(lsgm) + torch.pow(mu, 2)
        b = 1 + lsgm
        kld = 0.5 * torch.sum(a - b, dim=1)
        return kld
