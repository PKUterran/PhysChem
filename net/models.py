from .layers import *
from net.config import ConfType


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
        n_hop = config['N_HOP']
        n_global = config['N_GLOBAL']
        message_type = config['MESSAGE_TYPE']
        union_type = config['UNION_TYPE']
        derivation_type = config['DERIVATION_TYPE']
        tau = config['TAU']
        dropout = config['DROPOUT']
        self.use_cuda = use_cuda

        self.conf_type = config['CONF_TYPE']
        self.need_derive = self.conf_type is not ConfType.NONE and self.conf_type is not ConfType.RDKIT

        self.initializer = Initializer(
            atom_dim=atom_dim,
            bond_dim=bond_dim,
            hv_dim=hv_dim,
            he_dim=he_dim,
            p_dim=p_dim,
            q_dim=q_dim,
            use_cuda=use_cuda
        )
        self.mp_kernel = ConfAwareMPNNKernel(
            hv_dim=hv_dim,
            he_dim=he_dim,
            mv_dim=mv_dim,
            me_dim=me_dim,
            p_dim=p_dim,
            q_dim=q_dim,
            hops=n_hop,
            use_cuda=use_cuda,
            dropout=dropout,
            message_type=message_type,
            union_type=union_type
        )
        if self.need_derive:
            self.drv_kernel = InformedDerivationKernel(
                hv_dim=hv_dim,
                he_dim=he_dim,
                p_dim=p_dim,
                q_dim=q_dim,
                tau=tau,
                use_cuda=use_cuda,
                dropout=dropout,
                derivation_type=derivation_type
            )
        self.fingerprint_gen = FingerprintGenerator(
            hm_dim=hm_dim,
            hv_dim=hv_dim * self.n_layer if self.need_derive else hv_dim,
            mm_dim=mm_dim,
            iteration=n_global,
            use_cuda=use_cuda,
            dropout=dropout
        )
        self.conformation_gen = ConformationGenerator(
            q_dim=q_dim,
            h_dims=[16],
            dropout=dropout
        )

    # def forward(self, atom_ftr: torch.Tensor, bond_ftr: torch.Tensor, massive: torch.Tensor,
    #             mask_matrices: MaskMatrices
    #             ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     hv_ftr, he_ftr, p_ftr, q_ftr = self.initializer(atom_ftr, bond_ftr, mask_matrices)
    #     for i in range(self.n_layer):
    #         t_hv_ftr, t_he_ftr = self.mp_kernel(hv_ftr, he_ftr, p_ftr, q_ftr, mask_matrices)
    #
    #         for j in range(self.n_iteration):
    #             p_ftr, q_ftr = self.drv_kernel(hv_ftr, he_ftr, massive, p_ftr, q_ftr, mask_matrices)
    #
    #         hv_ftr, he_ftr = t_hv_ftr, t_he_ftr
    #
    #     fingerprint = self.fingerprint_gen(hv_ftr, p_ftr, q_ftr, mask_matrices)
    #     conformation = self.conformation_gen(q_ftr)
    #     return fingerprint, conformation

    def forward(self, atom_ftr: torch.Tensor, bond_ftr: torch.Tensor, massive: torch.Tensor,
                mask_matrices: MaskMatrices,
                given_p_ftr: torch.Tensor = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        hv_ftr, he_ftr, p_ftr, q_ftr = self.initializer.forward(atom_ftr, bond_ftr, mask_matrices)
        hv_ftrs = []

        if self.conf_type is ConfType.NONE:
            p_ftr, q_ftr = p_ftr * 0, q_ftr * 0
        elif self.conf_type is ConfType.RDKIT:
            p_ftr, q_ftr = given_p_ftr, q_ftr * 0

        for i in range(self.n_layer):
            t_hv_ftr, t_he_ftr = self.mp_kernel.forward(hv_ftr, he_ftr, p_ftr, q_ftr, mask_matrices)

            if self.need_derive:
                for j in range(self.n_iteration):
                    p_ftr, q_ftr = self.drv_kernel.forward(hv_ftr, he_ftr, massive, p_ftr, q_ftr, mask_matrices)

            hv_ftrs.append(t_hv_ftr)
            if not self.need_derive:
                break

        fingerprint = self.fingerprint_gen.forward(torch.cat(hv_ftrs, dim=1), mask_matrices)
        conformation = self.conformation_gen.forward(q_ftr)
        return fingerprint, conformation


class RecGeomNN(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int, config: dict,
                 use_cuda=False):
        super(RecGeomNN, self).__init__()
        hv_dim = config['HV_DIM']
        he_dim = config['HE_DIM']
        hm_dim = config['HM_DIM']
        mv_dim = config['MV_DIM']
        me_dim = config['ME_DIM']
        mm_dim = config['MM_DIM']
        p_dim = q_dim = config['PQ_DIM']
        self.n_iteration = config['N_ITERATION']
        n_hop = config['N_HOP']
        n_global = config['N_GLOBAL']
        message_type = config['MESSAGE_TYPE']
        union_type = config['UNION_TYPE']
        derivation_type = config['DERIVATION_TYPE']
        tau = config['TAU']
        dropout = config['DROPOUT']
        self.use_cuda = use_cuda

        self.conf_type = config['CONF_TYPE']
        self.need_derive = self.conf_type is not ConfType.NONE and self.conf_type is not ConfType.RDKIT

        self.initializer = Initializer(
            atom_dim=atom_dim,
            bond_dim=bond_dim,
            hv_dim=hv_dim,
            he_dim=he_dim,
            p_dim=p_dim,
            q_dim=q_dim,
            use_cuda=use_cuda
        )
        self.mp_kernel = ConfAwareMPNNKernel(
            hv_dim=hv_dim,
            he_dim=he_dim,
            mv_dim=mv_dim,
            me_dim=me_dim,
            p_dim=p_dim,
            q_dim=q_dim,
            hops=n_hop,
            use_cuda=use_cuda,
            dropout=dropout,
            message_type=message_type,
            union_type=union_type
        )
        if self.need_derive:
            self.drv_kernel = InformedDerivationKernel(
                hv_dim=hv_dim,
                he_dim=he_dim,
                p_dim=p_dim,
                q_dim=q_dim,
                tau=tau,
                use_cuda=use_cuda,
                dropout=dropout,
                derivation_type=derivation_type
            )
        self.fingerprint_gen = FingerprintGenerator(
            hm_dim=hm_dim,
            hv_dim=hv_dim,
            mm_dim=mm_dim,
            iteration=n_global,
            use_cuda=use_cuda,
            dropout=dropout
        )

    def forward(self, atom_ftr: torch.Tensor, bond_ftr: torch.Tensor, massive: torch.Tensor,
                mask_matrices: MaskMatrices,
                given_q_ftr: torch.Tensor = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        hv_ftr, he_ftr, p_ftr, q_ftr = self.initializer.forward(atom_ftr, bond_ftr, mask_matrices)

        if self.conf_type is ConfType.NONE:
            p_ftr, q_ftr = p_ftr * 0, q_ftr * 0
        elif self.conf_type is ConfType.RDKIT:
            p_ftr, q_ftr = p_ftr * 0, given_q_ftr

        for i in range(self.n_iteration):
            t_hv_ftr, t_he_ftr = self.mp_kernel.forward(hv_ftr, he_ftr, p_ftr, q_ftr, mask_matrices)

            if self.need_derive:
                p_ftr, q_ftr = self.drv_kernel.forward(hv_ftr, he_ftr, massive, p_ftr, q_ftr, mask_matrices)

            hv_ftr, he_ftr = t_hv_ftr, t_he_ftr

        fingerprint = self.fingerprint_gen.forward(hv_ftr, mask_matrices)
        conformation = q_ftr
        return fingerprint, conformation
