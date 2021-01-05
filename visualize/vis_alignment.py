import numpy as np
import torch
from typing import List, Dict, Tuple

from data.encode import encode_smiles, get_massive_from_atom_features
from net.utils.MaskMatrices import MaskMatrices, cuda_copy
from net.models import GeomNN
from train.utils.cache_batch import BatchCache
from .rebuild import rebuild_qm9


def generate_alignments(model: GeomNN, mol_info: Dict[str, np.ndarray]
                        ) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
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
    _, _, local_alignments, global_alignments = model.forward(atom_ftr, bond_ftr, massive, mask_matrices,
                                                              return_local_alignment=True, return_global_alignment=True)
    return local_alignments, global_alignments


def ve_align2vv_align(align: np.ndarray):
    n_vertex = align.shape[0]
    n_edge = int(align.shape[1] / 2)
    m = np.max(align, axis=0)
    am = np.argmax(align, axis=0)

    ijs = zip(am[:n_edge], am[n_edge:])
    vv = np.zeros([n_vertex, n_vertex], dtype=np.float)
    for t, (i, j) in enumerate(ijs):
        vv[i, j] = m[t]
        vv[j, i] = m[t + n_edge]
    return vv


def vis_alignment(list_smiles: List[str], tag: str, special_config: dict, use_cuda=False):
    np.set_printoptions(suppress=True, precision=3, linewidth=200)
    mols_info = encode_smiles(np.array(list_smiles, dtype=np.str))
    atom_dim, bond_dim = mols_info[0]['af'].shape[1], mols_info[0]['bf'].shape[1]
    model, classifier = rebuild_qm9(atom_dim, bond_dim, tag, special_config, use_cuda)
    for mol_info in mols_info:
        local_alignments, global_alignment = generate_alignments(model, mol_info)
        for i in range(len(local_alignments)):
            for j in range(len(local_alignments[i])):
                print(f'local: {i}, {j}')
                print(ve_align2vv_align(local_alignments[i][j]))
        for i in range(len(global_alignment)):
            print(f'global: {i}')
            print(global_alignment[i])
