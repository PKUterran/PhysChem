import os
import pickle
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from typing import Tuple, List
from rdkit.Chem.rdchem import Mol as Molecule
from data.config import QM9_CSV_PATH, QM9_SDF_PATH, QM9_PICKLE_PATH


def dump_qm8():
    supplier = Chem.SDMolSupplier(QM9_SDF_PATH)
    mols = list(supplier)
    not_none_mask = [i for i in range(len(mols)) if mols[i] is not None]
    df = pd.read_csv(QM9_CSV_PATH)
    csv: np.ndarray = df.values
    properties = csv[:, 1: 13].astype(np.float)
    mols = [mols[i] for i in not_none_mask]
    properties = properties[not_none_mask, :]
    with open(QM9_PICKLE_PATH, 'wb+') as fp:
        pickle.dump((mols, properties), fp)


def load_qm8(max_num=-1) -> Tuple[List[Molecule], np.ndarray]:
    if not os.path.exists(QM9_PICKLE_PATH):
        dump_qm8()
    with open(QM9_PICKLE_PATH, 'rb') as fp:
        mols, properties = pickle.load(fp)
    if 0 < max_num < len(mols):
        mols = mols[: max_num]
        properties = properties[: max_num, :]
    return mols, properties
