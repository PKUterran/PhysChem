import torch
import torch.nn as nn

from .layers import *


class GeomNN(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int, config: dict,
                 use_cuda=False):
        super(GeomNN, self).__init__()
        hv_dim = config['HV_DIM']
        he_dim = config['HE_DIM']
        hm_dim = config['HM_DIM']
        mv_dim = config['MV_DIM']
        me_dim = config['ME_DIM']
        mm_dim = config['MM_DIM']
        p_dim = q_dim = config['PQ_DIM']
        self.n_layer = config['N_LAYER']
        self.n_iteration = config['N_ITERATION']
        n_global = config['N_GLOBAL']
        message_type = config['MESSAGE_TYPE']
        union_type = config['UNION_TYPE']
        tau = config['TAU']
        dropout = config['DROPOUT']
        self.use_cuda = use_cuda

        self.initializer = Initializer(
            atom_dim=atom_dim,
            bond_dim=bond_dim,
            hv_dim=hv_dim,
            he_dim=he_dim,
            p_dim=p_dim,
            q_dim=q_dim,
            use_cuda=use_cuda
        )
        self.mp_kernels = nn.ModuleList([ConfAwareMPNNKernel(
            hv_dim=hv_dim,
            he_dim=he_dim,
            mv_dim=mv_dim,
            me_dim=me_dim,
            p_dim=p_dim,
            q_dim=q_dim,
            use_cuda=use_cuda,
            message_type=message_type,
            union_type=union_type
        ) for _ in range(self.n_layer)])
        self.ham_kernels = nn.ModuleList([InformedHamiltonianKernel(
            hv_dim=hv_dim,
            he_dim=he_dim,
            p_dim=p_dim,
            q_dim=q_dim,
            tau=tau,
            use_cuda=use_cuda,
            dropout=dropout
        ) for _ in range(self.n_layer)])
        self.fingerprint_gen = ConfAwareFingerprintGenerator(
            hm_dim=hm_dim,
            hv_dim=hv_dim,
            mm_dim=mm_dim,
            p_dim=p_dim,
            q_dim=q_dim,
            iteration=n_global,
            use_cuda=use_cuda
        )
        self.conformation_gen = ConformationGenerator(
            q_dim=q_dim,
            h_dims=[128],
            dropout=dropout
        )

    def forward(self, atom_ftr: torch.Tensor, bond_ftr: torch.Tensor, massive: torch.Tensor,
                mask_matrices: MaskMatrices
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        hv_ftr, he_ftr, p_ftr, q_ftr = self.initializer(atom_ftr, bond_ftr, mask_matrices)
        for i in range(self.n_layer):
            t_hv_ftr, t_he_ftr = self.mp_kernels[i](hv_ftr, he_ftr, p_ftr, q_ftr, mask_matrices)

            for j in range(self.n_iteration):
                p_ftr, q_ftr = self.ham_kernels[i](hv_ftr, he_ftr, massive, p_ftr, q_ftr, mask_matrices)

            hv_ftr, he_ftr = t_hv_ftr, t_he_ftr

        fingerprint = self.fingerprint_gen(hv_ftr, p_ftr, q_ftr, mask_matrices)
        conformation = self.conformation_gen(q_ftr)
        return fingerprint, conformation
