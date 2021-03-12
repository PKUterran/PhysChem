import os
import pickle
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from typing import Tuple, List
from rdkit.Chem.rdchem import Mol as Molecule
from data.config import QM9_CSV_PATH, QM9_SDF_PATH, QM9_PICKLE_PATH


def dump_qm9():
    supplier = Chem.SDMolSupplier(QM9_SDF_PATH)
    mols = [m for m in list(supplier) if m is not None and m.GetProp("_Name").startswith("gdb")]
    indices = [int(m.GetProp("_Name")[4:]) - 1 for m in mols]
    df = pd.read_csv(QM9_CSV_PATH)
    csv: np.ndarray = df.values
    properties = csv[:, 4: 16].astype(np.float)
    properties = properties[indices, :]
    with open(QM9_PICKLE_PATH, 'wb+') as fp:
        pickle.dump((mols, properties), fp)


def load_qm9(max_num=-1) -> Tuple[List[Molecule], np.ndarray]:
    if not os.path.exists(QM9_PICKLE_PATH):
        dump_qm9()
    with open(QM9_PICKLE_PATH, 'rb') as fp:
        mols, properties = pickle.load(fp)
    if 0 < max_num < len(mols):
        mols = mols[: max_num]
        properties = properties[: max_num, :]
    return mols, properties
