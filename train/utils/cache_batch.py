import os
import pickle
import torch
import numpy as np

from typing import List, Dict, Tuple, Any, Union
from tqdm import tqdm

from data.encode import get_massive_from_atom_features, encode_mols
from net.utils.MaskMatrices import MaskMatrices, cuda_copy
from train.utils.rdkit import rdkit_mol_positions

CACHE_DIR = 'train/utils/cache'
MOLS_DIR = 'train/utils/mols'


def get_mol_positions(mol) -> np.ndarray:
    return mol.GetConformer().GetPositions()


class Batch:
    def __init__(self, atom_ftr: torch.Tensor, bond_ftr: torch.Tensor, massive: torch.Tensor,
                 mask_matrices: MaskMatrices,
                 properties: torch.Tensor = None, conformation: torch.Tensor = None,
                 rdkit_conf: torch.Tensor = None):
        self.n_atom = atom_ftr.shape[0]
        self.n_bond = bond_ftr.shape[0]
        self.n_mol = mask_matrices.mol_vertex_w.shape[0]

        self.atom_ftr = atom_ftr
        self.bond_ftr = bond_ftr
        self.massive = massive
        self.mask_matrices = mask_matrices
        self.properties = properties
        self.conformation = conformation
        self.rdkit_conf = rdkit_conf


def batch_cuda_copy(batch: Batch) -> Batch:
    return Batch(
        atom_ftr=batch.atom_ftr.cuda(),
        bond_ftr=batch.bond_ftr.cuda(),
        massive=batch.massive.cuda(),
        mask_matrices=cuda_copy(batch.mask_matrices) if batch.mask_matrices is not None else None,
        properties=batch.properties.cuda() if batch.properties is not None else None,
        conformation=batch.conformation.cuda() if batch.conformation is not None else None,
        rdkit_conf=batch.rdkit_conf.cuda() if batch.rdkit_conf is not None else None,
    )


class BatchCache:
    def __init__(self, mols: List[Any], mols_info: List[Dict[str, np.ndarray]], mol_properties: np.ndarray,
                 needs_rdkit_conf=False, contains_ground_truth_conf=True, need_mask_matrices=True,
                 use_cuda=False, batch_size=32,
                 use_tqdm=False):
        assert len(mols_info) == mol_properties.shape[0]
        self.atom_dim = mols_info[0]['af'].shape[1]
        self.bond_dim = mols_info[0]['bf'].shape[1]
        self.n_mol = len(mols)

        self.mols = mols
        self.mols_info = mols_info
        self.mol_properties = mol_properties
        self.needs_rdkit_conf = needs_rdkit_conf
        self.contains_ground_truth_conf = contains_ground_truth_conf
        self.need_mask_matrices = need_mask_matrices
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
        self.train_masks: List[List[int]] = [train_mask[i::train_sep] for i in range(train_sep) if i < len(train_mask)]
        self.validate_masks: List[List[int]] = [validate_mask[i::validate_sep] for i in range(validate_sep)
                                                if i < len(validate_mask)]
        self.test_masks: List[List[int]] = [test_mask[i::test_sep] for i in range(test_sep) if i < len(test_mask)]

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
            if sum(n_bonds) == 0:
                continue
            ms = []
            us = []
            vs = []
            for i, m in enumerate(mask):
                ms.extend([i] * n_atoms[i])
                prev_bonds = sum(n_atoms[:i])
                us.extend(self.mols_info[m]['us'] + prev_bonds)
                vs.extend(self.mols_info[m]['vs'] + prev_bonds)

            properties = self.mol_properties[mask, :].astype(np.float32)
            atom_ftr = torch.from_numpy(atom_ftr).type(torch.float32)
            bond_ftr = torch.from_numpy(bond_ftr).type(torch.float32)
            massive = torch.from_numpy(massive).type(torch.float32)
            properties = torch.from_numpy(properties).type(torch.float32)

            if self.contains_ground_truth_conf:
                conformation = np.vstack([get_mol_positions(self.mols[m]) for m in mask])
                assert conformation.shape[0] == sum(n_atoms)
                conformation = torch.from_numpy(conformation).type(torch.float32)
            else:
                conformation = None
            if self.needs_rdkit_conf:
                rdkit_conf = np.vstack([rdkit_mol_positions(self.mols[m]) for m in mask])
                assert rdkit_conf.shape[0] == sum(n_atoms)
                rdkit_conf = torch.from_numpy(rdkit_conf).type(torch.float32)
                if not self.contains_ground_truth_conf:
                    conformation = rdkit_conf
            else:
                rdkit_conf = None

            if self.need_mask_matrices:
                mol_vertex_w, mol_vertex_b = self.produce_mask_matrix(len(mask), ms)
                vertex_edge_w1, vertex_edge_b1 = self.produce_mask_matrix(sum(n_atoms), us)
                vertex_edge_w2, vertex_edge_b2 = self.produce_mask_matrix(sum(n_atoms), vs)
                mol_vertex_w = torch.from_numpy(mol_vertex_w).type(torch.float32)
                mol_vertex_b = torch.from_numpy(mol_vertex_b).type(torch.float32)
                vertex_edge_w1 = torch.from_numpy(vertex_edge_w1).type(torch.float32)
                vertex_edge_b1 = torch.from_numpy(vertex_edge_b1).type(torch.float32)
                vertex_edge_w2 = torch.from_numpy(vertex_edge_w2).type(torch.float32)
                vertex_edge_b2 = torch.from_numpy(vertex_edge_b2).type(torch.float32)
                mask_matrices = MaskMatrices(mol_vertex_w, mol_vertex_b,
                                             vertex_edge_w1, vertex_edge_w2,
                                             vertex_edge_b1, vertex_edge_b2)
            else:
                mask_matrices = None

            batch = Batch(atom_ftr, bond_ftr, massive, mask_matrices, properties, conformation, rdkit_conf)
            batches.append(batch)

        return batches

    @staticmethod
    def produce_mask_matrix(n: int, s: list) -> Tuple[np.ndarray, np.ndarray]:
        s = np.array(s)
        mat = np.full([n, s.shape[0]], 0., dtype=np.int32)
        mask = np.full([n, s.shape[0]], -1e6, dtype=np.int32)
        for i in range(n):
            node_edge = s == i
            mat[i, node_edge] = 1
            mask[i, node_edge] = 0
        return mat, mask


def load_batch_cache(name: str, mols: List[Any], mols_info: List[Dict[str, np.ndarray]], mol_properties: np.ndarray,
                     needs_rdkit_conf=False, contains_ground_truth_conf=True, need_mask_matrices=True,
                     use_cuda=False, batch_size=32,
                     force_save=False, use_tqdm=False) -> BatchCache:
    if not os.path.exists(CACHE_DIR):
        os.mkdir(CACHE_DIR)
    pickle_path = f'{CACHE_DIR}/{name}.pickle'
    if not os.path.exists(pickle_path) or force_save:
        print('\tProducing New Batches...')
        batch_cache = BatchCache(mols, mols_info, mol_properties,
                                 needs_rdkit_conf=needs_rdkit_conf,
                                 contains_ground_truth_conf=contains_ground_truth_conf,
                                 need_mask_matrices=need_mask_matrices,
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


def produce_batches_from_mols(mols: List[Any]) -> List[Batch]:
    mols_info = encode_mols(mols)
    batches = []
    for mol_info in mols_info:
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

        batches.append(Batch(
            atom_ftr=atom_ftr,
            bond_ftr=bond_ftr,
            massive=massive,
            mask_matrices=mask_matrices,
        ))

    return batches
