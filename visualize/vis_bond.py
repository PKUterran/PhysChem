import numpy as np
import torch
from tqdm import tqdm
from rdkit import Chem
from typing import List, Dict, Tuple

from data.encode import encode_mols, get_massive_from_atom_features
from net.utils.MaskMatrices import MaskMatrices
from net.models import GeomNN
from train.utils.cache_batch import BatchCache
from .rebuild import rebuild_qm9
from .bond_energy import get_actual_bond_energy
from .bond.plt_bond import plt_predict_actual_bond_energy


def generate_bond_energy(model: GeomNN, mol_info: Dict[str, np.ndarray]) -> List[np.ndarray]:
    af, bf, us, vs = mol_info['af'], mol_info['bf'], mol_info['us'], mol_info['vs']
    massive = get_massive_from_atom_features(af)
    mvw, mvb = BatchCache.produce_mask_matrix(1, [0] * af.shape[0])
    vew1, veb1 = BatchCache.produce_mask_matrix(af.shape[0], list(us))
    vew2, veb2 = BatchCache.produce_mask_matrix(af.shape[0], list(vs))

    atom_ftr = torch.from_numpy(af).type(torch.float32)
    bond_ftr = torch.from_numpy(bf).type(torch.float32)
    massive = torch.from_numpy(massive).type(torch.float32)
    mol_vertex_w = torch.from_numpy(mvw).type(torch.float32)
    mol_vertex_b = torch.from_numpy(mvb).type(torch.float32)
    vertex_edge_w1 = torch.from_numpy(vew1).type(torch.float32)
    vertex_edge_b1 = torch.from_numpy(veb1).type(torch.float32)
    vertex_edge_w2 = torch.from_numpy(vew2).type(torch.float32)
    vertex_edge_b2 = torch.from_numpy(veb2).type(torch.float32)

    mask_matrices = MaskMatrices(mol_vertex_w, mol_vertex_b,
                                 vertex_edge_w1, vertex_edge_w2,
                                 vertex_edge_b1, vertex_edge_b2)
    _, _, _, _, list_he_ftr = model.forward(atom_ftr, bond_ftr, massive, mask_matrices)
    return [np.sqrt(np.sum(he_ftr ** 2, axis=1)) for he_ftr in list_he_ftr]


def vis_bond(list_smiles: List[str], tag: str, special_config: dict, use_cuda=False):
    n_smiles = len(list_smiles)
    list_mol = [Chem.MolFromSmiles(s) for s in list_smiles]
    mols_info = encode_mols(list_mol)
    atom_dim, bond_dim = mols_info[0]['af'].shape[1], mols_info[0]['bf'].shape[1]
    model, classifier = rebuild_qm9(atom_dim, bond_dim, tag, special_config, use_cuda)
    n_layer = model.n_layer
    n1 = [[] for _ in range(n_layer)]
    n2 = [[] for _ in range(n_layer)]
    actual = []
    t = tqdm(range(n_smiles), total=n_smiles)
    for idx in t:
        pbes = generate_bond_energy(model, mols_info[idx])
        abe = get_actual_bond_energy(list_mol[idx])
        mask = abe > 0
        for j in range(n_layer):
            pbe = pbes[j]
            n1[j].extend(pbe[mask])
            n2[j].extend(pbe[mask] ** 2)
        actual.extend(abe[mask])
    for j in range(n_layer):
        plt_predict_actual_bond_energy(n1[j], actual, title=f'N1-{j}')
        plt_predict_actual_bond_energy(n2[j], actual, title=f'N2-{j}')
