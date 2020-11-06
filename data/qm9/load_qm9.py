import json
import pickle
import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem.rdchem import Mol as Molecule
from rdkit.Chem.rdmolops import RemoveAllHs
from typing import Tuple, List

from data.config import QM9_RDKIT_SUMMARY_PATH, RDKIT_FOLDER_DIR, QM9_CSV2JSON_PATH, QM9_CSV_PATH, QM9_PICKLE_PATH

WEIGHT_GATE = 0.2


def get_mol_positions(mol: Molecule) -> np.ndarray:
    return mol.GetConformers()[0].GetPositions()


def mol_pickle2list(p: dict) -> List[Tuple[float, Molecule]]:
    confs = p['conformers']
    weights = []
    mols = []
    for conf in confs:
        weight = conf['boltzmannweight']
        if weight < WEIGHT_GATE:
            continue

        mol = conf['rd_mol']
        mol = RemoveAllHs(mol)
        weights.append(weight)
        mols.append(mol)

    s = sum(weights)
    weights = [w / s for w in weights]
    return list(zip(weights, mols))


def cache_qm9():
    print('Loading QM9...')
    df = pd.read_csv(QM9_CSV_PATH)
    csv: np.ndarray = df.values

    o_keys = csv[:, 1]
    o_values = csv[:, -12:]

    fp = open(QM9_CSV2JSON_PATH)
    csv2json: dict = json.load(fp)
    fp.close()
    fp = open(QM9_RDKIT_SUMMARY_PATH)
    summary = json.load(fp)
    fp.close()

    r_keys = [csv2json.setdefault(k, '') for k in o_keys]
    keys = []
    for k in r_keys:
        if k != '' and 'pickle_path' in summary[k]:
            keys.append(k)
        else:
            keys.append('')
    keys = np.array(keys, dtype=np.str)
    mask = keys != ''
    print('\tAvailable: {:.2f}%'.format(100 * sum(mask) / len(mask)))
    keys = keys[mask]
    mol_properties: np.ndarray = o_values[mask, :]

    mol_list_weight_mol = []
    for i, k in enumerate(keys):
        path = summary[k]["pickle_path"]
        fp = open(f'{RDKIT_FOLDER_DIR}/{path}', 'rb')
        p: dict = pickle.load(fp)
        fp.close()
        mol_list_weight_mol.append(mol_pickle2list(p))
        if (i + 1) % 1000 == 0:
            print('\t{}/{} loaded'.format(i + 1, sum(mask)))

    assert len(mol_list_weight_mol) == mol_properties.shape[0]
    print('\tCaching QM9...')
    fp = open(QM9_PICKLE_PATH, 'wb+')
    pickle.dump((mol_list_weight_mol, mol_properties), fp)
    fp.close()
    print('\tCaching Finished!')


def load_qm9(max_num: int = -1) -> Tuple[List[List[Tuple[float, Molecule]]], np.ndarray]:
    print('\tLoading QM9...')
    fp = open(QM9_PICKLE_PATH, 'rb')
    mol_list_weight_mol, mol_properties = pickle.load(fp)
    fp.close()
    assert len(mol_list_weight_mol) == mol_properties.shape[0]
    print('\tLoading Finished!')
    if max_num != -1:
        assert max_num <= len(mol_list_weight_mol), \
            f'{len(mol_list_weight_mol)} smiles in QM9 while {max_num} are required'
        mol_list_weight_mol = mol_properties[: max_num]
        mol_properties = mol_properties[:max_num, :]

    return mol_list_weight_mol, mol_properties
