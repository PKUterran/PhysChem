import torch
import torch.nn as nn

from .components import *


class Initializer(nn.Module):
    def __init__(self, ):
        super(Initializer, self).__init__()

    def forward(self, *input: Any, **kwargs: Any) -> T_co:
        pass


class ConfAwareMPNNKernel(nn.Module):
    def __init__(self, hv_dim: int, he_dim: int, mv_dim: int, me_dim: int, p_dim: int, q_dim: int,
                 use_cuda=False, message_type='naive', union_type='gru'):
        super(ConfAwareMPNNKernel, self).__init__()
        self.use_cuda = use_cuda
        self.message_type = message_type
        self.union_type = union_type

        if message_type == 'naive':
            self.message = NaiveDynMessage(hv_dim, he_dim, mv_dim, me_dim, p_dim, q_dim, use_cuda)
        else:
            assert False, 'Undefined message type {} in net.layers.ConfAwareMPNNKernel'.format(message_type)

        if union_type == 'naive':
            self.union_v = NaiveUnion(hv_dim, mv_dim, use_cuda)
            self.union_e = NaiveUnion(he_dim, me_dim, use_cuda)
        elif union_type == 'gru':
            self.union_v = GRUUnion(hv_dim, mv_dim, use_cuda)
            self.union_e = GRUUnion(he_dim, me_dim, use_cuda)
        else:
            assert False, 'Undefined union type {} in net.layers.ConfAwareMPNNKernel'.format(union_type)

    def forward(self, hv_ftr: torch.Tensor, he_ftr: torch.Tensor, p_ftr: torch.Tensor, q_ftr: torch.Tensor,
                mask_matrices: MaskMatrices
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        mv_ftr, me_ftr = self.message(hv_ftr, he_ftr, p_ftr, q_ftr, mask_matrices)
        hv_ftr = self.union_v(hv_ftr, mv_ftr)
        he_ftr = self.union_e(he_ftr, me_ftr)
        return hv_ftr, he_ftr


class InformedHamiltonianKernel(nn.Module):
    def __init__(self):
        super(InformedHamiltonianKernel, self).__init__()

    def forward(self, *input: Any, **kwargs: Any) -> T_co:
        pass


class ConfAwareRepresentation(nn.Module):
    def __init__(self, hm_dim: int, hv_dim: int, mm_dim: int, p_dim: int, q_dim: int, iteration: int,
                 use_cuda=False):
        super(ConfAwareRepresentation, self).__init__()
        self.use_cuda = use_cuda

        self.vertex2mol = nn.Linear(hv_dim, hm_dim, bias=True)
        self.vm_act = nn.LeakyReLU()
        self.readout = GlobalDynReadout(hm_dim, hv_dim, mm_dim, p_dim, q_dim, use_cuda)
        self.union = GRUUnion(hm_dim, mm_dim, use_cuda)
        self.iteration = iteration

    def forward(self, hv_ftr: torch.Tensor, p_ftr: torch.Tensor, q_ftr: torch.Tensor,
                mask_matrices: MaskMatrices
                ) -> torch.Tensor:
        # initialize molecule features with mean of vertex features
        mvw = mask_matrices.mol_vertex_w
        norm_mvw = mvw / torch.sum(mvw, dim=-1, keepdim=True)
        hm_ftr = norm_mvw @ self.vm_act(self.vertex2mol(hv_ftr))

        # iterate
        for i in range(self.iteration):
            mm_ftr = self.readout(hm_ftr, hv_ftr, p_ftr, q_ftr, mask_matrices)
            hm_ftr = self.union(hm_ftr, mm_ftr)

        return hm_ftr


class ConformationGenerator(nn.Module):
    def __init__(self):
        super(ConformationGenerator, self).__init__()

    def forward(self, *input: Any, **kwargs: Any) -> T_co:
        pass
