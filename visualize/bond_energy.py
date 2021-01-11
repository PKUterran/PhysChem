import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import Mol as Molecule

SINGLE = Chem.rdchem.BondType.SINGLE
DOUBLE = Chem.rdchem.BondType.DOUBLE
TRIPLE = Chem.rdchem.BondType.TRIPLE
AROMATIC = Chem.rdchem.BondType.AROMATIC

BE_DICT = {
    # 'C-C': 332,
    'C|C': 518,
    'C=C': 611,
    'C#C': 837,
    'C-F': 485,
    # 'C-N': 305,
    'C|N': 504,
    'C=N': 615,
    'C#N': 891,
    # 'C-O': 326,
    'C|O': 556,
    'C=O': 728,
    'C-S': 272,
    'C=S': 536,
    'N-N': 156,
    'N|N': 334,
    'N=N': 456,
    'N#N': 946,
    'N-O': 230,
    'N|O': 478,
    'N=O': 607,
    'O-O': 146,
    'O=O': 498,
    'F-F': 153,
    'S-O': 364,
    'S-S': 268,
}


def __fetch(symbol1: str, symbol2: str, bond_type: int) -> float:
    if bond_type == SINGLE:
        bt_symbol = '-'
    elif bond_type == DOUBLE:
        bt_symbol = '='
    elif bond_type == TRIPLE:
        bt_symbol = '#'
    elif bond_type == AROMATIC:
        bt_symbol = '|'
    else:
        assert False, f'{bond_type}'
    k1 = symbol1 + bt_symbol + symbol2
    k2 = symbol2 + bt_symbol + symbol1
    if k1 in BE_DICT.keys():
        be = BE_DICT[k1]
    elif k2 in BE_DICT.keys():
        be = BE_DICT[k2]
    else:
        # print(f'Neither {k1} nor {k2} in BE_DICT')
        be = 0
    return be


def get_actual_bond_energy(mol: Molecule) -> np.ndarray:
    ret = []
    for b in mol.GetBonds():
        ret.append(__fetch(b.GetBeginAtom().GetSymbol(),
                           b.GetEndAtom().GetSymbol(),
                           b.GetBondType()))
    return np.array(ret, dtype=np.float)
