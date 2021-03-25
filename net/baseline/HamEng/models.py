import torch
import torch.nn as nn

from .layers import LstmPQEncoder
from net.utils.MaskMatrices import MaskMatrices
from net.dynamics.hamiltion import DissipativeHamiltonianDerivation


class HamiltonianPositionProducer(nn.Module):
    def __init__(self, n_dim, e_dim, config, use_cuda=True):
        super(HamiltonianPositionProducer, self).__init__()
        self.p_dim = config['PQ_DIM']
        self.q_dim = config['PQ_DIM']
        self.layers = config['HGN_LAYERS']
        self.tau = config['TAU']
        self.dropout = config['DROPOUT']
        self.dissipate = config['DISSIPATE']
        self.disturb = config['DISTURB']
        self.use_lstm = config['LSTM']
        self.use_cuda = use_cuda

        self.e_encoder = nn.Linear(n_dim + e_dim + n_dim, 1)
        self.pq_encoder = LstmPQEncoder(n_dim, self.p_dim,
                                        use_cuda=use_cuda, disturb=self.disturb, use_lstm=self.use_lstm)
        self.derivation = DissipativeHamiltonianDerivation(n_dim, e_dim, self.p_dim, self.q_dim,
                                                           use_cuda=use_cuda, dropout=0.0)

    def forward(self, v_features: torch.Tensor, e_features: torch.Tensor, massive: torch.Tensor,
                mask_matrices: MaskMatrices, return_multi=False):
        mvw = mask_matrices.mol_vertex_w
        vew1 = mask_matrices.vertex_edge_w1
        vew2 = mask_matrices.vertex_edge_w2
        u_e_v_features = torch.cat([vew1.t() @ v_features, e_features, vew2.t() @ v_features], dim=1)
        e_weight = torch.diag(torch.sigmoid(self.e_encoder(u_e_v_features)).view([-1]))
        e = vew1 @ e_weight @ vew2.t()
        p0, q0 = self.pq_encoder(v_features, mvw, e)
        ps = [p0]
        qs = [q0]
        s_losses = []
        c_losses = []
        h = None
        d = None

        for i in range(self.layers):
            dp, dq, h, d = self.derivation.forward(v_features, e_features, massive, ps[i], qs[i], mask_matrices,
                                                   return_energy=True, dissipate=self.dissipate)
            ps.append(ps[i] + self.tau * dp)
            qs.append(qs[i] + self.tau * dq)

            s_losses.append((dq - ps[i]).norm())
            c_losses.append((mvw @ (ps[i + 1] - ps[i])).norm())

        s_loss = sum(s_losses)
        c_loss = sum(c_losses)
        if self.dissipate:
            final_p = ps[-1]
            final_q = qs[-1]
        else:
            final_p = sum(ps) / len(ps)
            final_q = sum(qs) / len(qs)

        if return_multi:
            return ps, qs, s_loss, c_loss, h, d
        return final_p, final_q, s_loss, c_loss, h, d
