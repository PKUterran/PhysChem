import pandas as pd
from rdkit import Chem

from data.qm8.load_qm8 import load_qm8
from data.qm9.load_qm9 import load_qm9
from net.config import ConfType
from visualize.eval_rmsd import eval_rmsd_with_mols

'''
scp 1500011335@115.27.161.31:yangshuwen/GeomNN/train/models/QM9-Oconf* train/models/
Available: 19633/20000
Guess: 1.852809122054676
RDKit: 0.7347384707025154
CVGAE: 1.2589760663587264
HamEng: 0.7194251172988463
GeomNN: 0.5809578502960406
GeomNN better than RDKit: 12204/19633
'''


def rmsd_script():
    mols, _ = load_qm9()
    # mols = mols[-5000:]
    print(len(mols))
    eval_rmsd_with_mols(
        list_mols=mols,
        tag='QM9-Oconf',
        special_config={
            'CONF_TYPE': ConfType.ONLY,
        },
    )


if __name__ == '__main__':
    rmsd_script()
