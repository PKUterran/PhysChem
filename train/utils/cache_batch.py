import os
import pickle
import torch
import numpy as np

from typing import List, Dict, Tuple, Any, Union
from tqdm import tqdm

from data.encode import get_massive_from_atom_features, encode_mols
from data.qm9.load_qm9 import get_mol_positions
from net.utils.MaskMatrices import MaskMatrices

CACHE_DIR = 'train/utils/cache'
MOLS_DIR = 'train/utils/mols'


class Batch:
    def __init__(self, atom_ftr: torch.Tensor, bond_ftr: torch.Tensor, massive: torch.Tensor,
                 mask_matrices: MaskMatrices,
                 properties: torch.Tensor = None, conformation: torch.Tensor = None):
        self.n_atom = atom_ftr.shape[0]
        self.n_bond = bond_ftr.shape[0]
        self.n_mol = mask_matrices.mol_vertex_w.shape[0]

        self.atom_ftr = atom_ftr
        self.bond_ftr = bond_ftr
        self.massive = massive
        self.mask_matrices = mask_matrices
        self.properties = properties
        self.conformation = conformation


class BatchCache:
    def __init__(self, mols: List[Any], mols_info: List[Dict[str, np.ndarray]], mol_properties: np.ndarray,
                 use_cuda=False, batch_size=32,
                 use_tqdm=False):
        assert len(mols_info) == mol_properties.shape[0]
        self.atom_dim = mols_info[0]['af'].shape[1]
        self.bond_dim = mols_info[0]['bf'].shape[1]
        self.n_mol = len(mols)

        self.mols = mols
        self.mols_info = mols_info
        self.mol_properties = mol_properties
        self.use_cuda = use_cuda
        self.use_tqdm = use_tqdm

        n_mol = len(mols_info)
        train_num = int(n_mol * 0.8)
        validate_num = int(n_mol * 0.1)
        test_num = n_mol - train_num - validate_num

        mask = np.random.permutation(n_mol)
        train_mask = mask[:train_num]
        validate_mask = mask[train_num: -test_num]
        test_mask = mask[-test_num:]

        train_sep = int(train_num / batch_size) + 1
        validate_sep = int(validate_num / batch_size) + 1
        test_sep = int(test_num / batch_size) + 1
        self.train_masks: List[List[int]] = [train_mask[i::train_sep] for i in range(train_sep)]
        self.validate_masks: List[List[int]] = [validate_mask[i::validate_sep] for i in range(validate_sep)]
        self.test_masks: List[List[int]] = [test_mask[i::test_sep] for i in range(test_sep)]

        print('\t\tProducing Train Batches:')
        self.train_batches: List[Batch] = self.produce_batches(self.train_masks)
        print('\t\tProducing Validate Batches:')
        self.validate_batches: List[Batch] = self.produce_batches(self.validate_masks)
        print('\t\tProducing Test Batches:')
        self.test_batches: List[Batch] = self.produce_batches(self.test_masks)

    def produce_batches(self, masks: List[List[int]]) -> List[Batch]:
        batches = []
        if self.use_tqdm:
            masks = tqdm(masks, total=len(masks))
        for mask in masks:
            atom_ftr = np.vstack([self.mols_info[m]['af'] for m in mask])
            bond_ftr = np.vstack([self.mols_info[m]['bf'] for m in mask])
            massive = get_massive_from_atom_features(atom_ftr)
            n_atoms = [self.mols_info[m]['af'].shape[0] for m in mask]
            n_bonds = [self.mols_info[m]['bf'].shape[0] for m in mask]
            ms = []
            us = []
            vs = []
            for i, m in enumerate(mask):
                ms.extend([i] * n_atoms[i])
                prev_bonds = sum(n_bonds[:i])
                us.extend(self.mols_info[m]['us'] + prev_bonds)
                vs.extend(self.mols_info[m]['vs'] + prev_bonds)

            mol_vertex_w, mol_vertex_b = self.produce_mask_matrix(len(mask), ms)
            vertex_edge_w1, vertex_edge_b1 = self.produce_mask_matrix(sum(n_atoms), us)
            vertex_edge_w2, vertex_edge_b2 = self.produce_mask_matrix(sum(n_atoms), vs)

            mp = self.mol_properties[mask, :].astype(np.float)
            mc = np.vstack([get_mol_positions(self.mols[m]) for m in mask])
            assert mc.shape[0] == sum(n_atoms)

            atom_ftr = torch.from_numpy(atom_ftr).type(torch.float32)
            bond_ftr = torch.from_numpy(bond_ftr).type(torch.float32)
            massive = torch.from_numpy(massive).type(torch.float32)
            mol_vertex_w = torch.from_numpy(mol_vertex_w).type(torch.float32)
            mol_vertex_b = torch.from_numpy(mol_vertex_b).type(torch.float32)
            vertex_edge_w1 = torch.from_numpy(vertex_edge_w1).type(torch.float32)
            vertex_edge_b1 = torch.from_numpy(vertex_edge_b1).type(torch.float32)
            vertex_edge_w2 = torch.from_numpy(vertex_edge_w2).type(torch.float32)
            vertex_edge_b2 = torch.from_numpy(vertex_edge_b2).type(torch.float32)
            mp = torch.from_numpy(mp).type(torch.float32)
            mc = torch.from_numpy(mc).type(torch.float32)
            if self.use_cuda:
                atom_ftr = atom_ftr.cuda()
                bond_ftr = bond_ftr.cuda()
                massive = massive.cuda()
                mol_vertex_w = mol_vertex_w.cuda()
                mol_vertex_b = mol_vertex_b.cuda()
                vertex_edge_w1 = vertex_edge_w1.cuda()
                vertex_edge_b1 = vertex_edge_b1.cuda()
                vertex_edge_w2 = vertex_edge_w2.cuda()
                vertex_edge_b2 = vertex_edge_b2.cuda()
                mp = mp.cuda()
                mc = mc.cuda()

            mask_matrices = MaskMatrices(mol_vertex_w, mol_vertex_b,
                                         vertex_edge_w1, vertex_edge_w2,
                                         vertex_edge_b1, vertex_edge_b2)
            batch = Batch(atom_ftr, bond_ftr, massive, mask_matrices, mp, mc)
            batches.append(batch)

        return batches

    @staticmethod
    def produce_mask_matrix(n: int, s: list) -> Tuple[np.ndarray, np.ndarray]:
        s = np.array(s)
        mat = np.full([n, s.shape[0]], 0., dtype=np.int)
        mask = np.full([n, s.shape[0]], -1e6, dtype=np.int)
        for i in range(n):
            node_edge = s == i
            mat[i, node_edge] = 1
            mask[i, node_edge] = 0
        return mat, mask


def load_batch_cache(name: str, mols: List[Any], mols_info: List[Dict[str, np.ndarray]], mol_properties: np.ndarray,
                     use_cuda=False, batch_size=32,
                     force_save=False, use_tqdm=False) -> BatchCache:
    if not os.path.exists(CACHE_DIR):
        os.mkdir(CACHE_DIR)
    pickle_path = f'{CACHE_DIR}/{name}.pickle'
    if not os.path.exists(pickle_path) or force_save:
        print('\tProducing New Batches...')
        batch_cache = BatchCache(mols, mols_info, mol_properties,
                                 use_cuda=use_cuda, batch_size=batch_size, use_tqdm=use_tqdm)
        with open(pickle_path, 'wb+') as fp:
            pickle.dump(batch_cache, fp)
    else:
        print('\tUse Cached Batches')
        with open(pickle_path, 'rb') as fp:
            batch_cache = pickle.load(fp)

    return batch_cache


def load_encode_mols(mols, name: str = None, force_save=False, return_mask=False
                     ) -> Union[List[Dict[str, np.ndarray]], Tuple[List[Dict[str, np.ndarray]], List[int]]]:
    if not os.path.exists(MOLS_DIR):
        os.mkdir(MOLS_DIR)
    if name is None:
        return encode_mols(mols, return_mask=return_mask)

    pickle_path = f'{MOLS_DIR}/{name}.pickle'
    if not os.path.exists(pickle_path) or force_save:
        ret = encode_mols(mols, return_mask=return_mask)
        with open(pickle_path, 'wb+') as fp:
            pickle.dump(ret, fp)
    else:
        print('\tUse Cached Mols')
        with open(pickle_path, 'rb') as fp:
            ret = pickle.load(fp)

    return ret
