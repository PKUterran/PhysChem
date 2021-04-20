import os
import pickle
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from typing import Tuple, List
from rdkit.Chem.rdchem import Mol as Molecule
from data.config import QM8_CSV_PATH, QM8_SDF_PATH, QM8_PICKLE_PATH


def dump_qm8():
    supplier = Chem.SDMolSupplier(QM8_SDF_PATH)
    mols = [m for m in supplier if m is not None and m.GetProp("_Name").startswith("gdb")]
    mols = [Chem.RemoveAllHs(mol) for mol in mols]
    indices = [int(m.GetProp("_Name")[4:].split('\t')[0]) for m in mols]
    i_m = zip(indices, mols)
    i_m = [(i, m) for i, m in i_m]
    mols = [m for _, m in i_m]
    indices = [i for i, _ in i_m]

    df = pd.read_csv(QM8_CSV_PATH)
    csv: np.ndarray = df.values
    p_indices = list(csv[:, 0])
    r_indices = []
    for idx in indices:
        r_indices.append(p_indices.index(idx))
    properties = csv[:, list(range(1, 17))].astype(np.float)
    properties = properties[r_indices, :]

    with open(QM8_PICKLE_PATH, 'wb+') as fp:
        pickle.dump((mols, properties), fp)


def load_qm8(max_num=-1) -> Tuple[List[Molecule], np.ndarray]:
    if not os.path.exists(QM8_PICKLE_PATH):
        dump_qm8()
    with open(QM8_PICKLE_PATH, 'rb') as fp:
        mols, properties = pickle.load(fp)
    if 0 < max_num < len(mols):
        mols = mols[: max_num]
        properties = properties[: max_num, :]
    return mols, properties
