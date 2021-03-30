import numpy as np
import torch
from typing import List, Dict, Tuple
from rdkit import Chem
from rdkit.Chem.rdchem import Mol as Molecule

from data.encode import encode_mols, get_massive_from_atom_features
from net.utils.MaskMatrices import MaskMatrices, cuda_copy
from net.models import GeomNN
from net.baseline.CVGAE.PredX_MPNN import CVGAE
from net.baseline.HamEng.models import HamiltonianPositionProducer
from train.utils.loss_functions import adj3_loss
from train.utils.cache_batch import BatchCache, get_mol_positions
from train.utils.rdkit import rdkit_mol_positions
from .rebuild import rebuild_qm9
from .derive.plt_derive import plt_derive, log_pos_json


def generate_derive(model: GeomNN, mol_info: Dict[str, np.ndarray]
                    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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
    # adj3_loss(None, None, mask_matrices, use_cuda=False)
    _, _, _, _, _, list_p_ftr, list_q_ftr = model.forward(atom_ftr, bond_ftr, massive, mask_matrices,
                                                          return_derive=True)
    return list_p_ftr, list_q_ftr


def vis_derive_with_smiles(list_smiles: List[str], tag: str, special_config: dict, use_cuda=False):
    np.set_printoptions(suppress=True, precision=3, linewidth=200)
    list_mols = [Chem.MolFromSmiles(list_smiles[i]) for i in range(len(list_smiles))]
    mols_info = encode_mols(list_mols)
    atom_dim, bond_dim = mols_info[0]['af'].shape[1], mols_info[0]['bf'].shape[1]
    model, classifier = rebuild_qm9(atom_dim, bond_dim, tag, special_config, use_cuda)
    for idx, mol_info in enumerate(mols_info):
        list_p, list_q = generate_derive(model, mol_info)
        conf = rdkit_mol_positions(Chem.MolFromSmiles(list_smiles[idx]))
        plt_derive(conf, None, list_mols[idx], f'm{idx}_rdkit')
        for t, (p, q) in enumerate(zip(list_p, list_q)):
            plt_derive(q, p, list_mols[idx], f'm{idx}_derive_{t}', d='visualize/derive/graph_smiles')


def vis_derive_with_mols(list_mols: List[Molecule], tag: str, special_config: dict, use_cuda=False):
    np.set_printoptions(suppress=True, precision=3, linewidth=200)
    list_smiles = [Chem.MolToSmiles(list_mols[i]) for i in range(len(list_mols))]
    mols_info = encode_mols(list_mols)
    atom_dim, bond_dim = mols_info[0]['af'].shape[1], mols_info[0]['bf'].shape[1]
    model, classifier = rebuild_qm9(atom_dim, bond_dim, tag, special_config, use_cuda)
    for idx, mol_info in enumerate(mols_info):
        list_p, list_q = generate_derive(model, mol_info)

        conf = get_mol_positions(list_mols[idx])
        log_pos_json(conf, None, list_mols[idx], list_smiles[idx], f'm{idx}_real')
        plt_derive(conf, None, list_mols[idx], f'm{idx}_real')

        conf = rdkit_mol_positions(list_mols[idx])
        log_pos_json(conf, None, list_mols[idx], list_smiles[idx], f'm{idx}_rdkit')
        plt_derive(conf, None, list_mols[idx], f'm{idx}_rdkit')

        for t, (p, q) in enumerate(zip(list_p, list_q)):
            log_pos_json(q, p, list_mols[idx], list_smiles[idx], f'm{idx}_derive_{t}')
            plt_derive(q, p, list_mols[idx], f'm{idx}_derive_{t}')
