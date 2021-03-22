from net.config import ConfType
from train.CVGAE.train_qm9 import train_qm9, QMDataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--pos', type=int, default=0)
arg = parser.parse_args()
seed = arg.seed
pos = arg.pos

if pos == 0:
    conf_type = ConfType.NONE
    name = 'QM7-Xconf'
else:
    conf_type = ConfType.RDKIT
    name = 'QM7-rdkit'

train_qm9(
    special_config={
        'HV_DIM': 128,
        'HE_DIM': 64,

        'EPOCH': 300,
        'BATCH': 50,
        'SAMPLE': 5,
        'LAMBDA': 1,
        'LR': 1e-5,
        'GAMMA': 0.995,
        'DECAY': 1e-5,

        'CONF_TYPE': ConfType.NEWTON,
    },
    dataset=QMDataset.QM7,
    use_cuda=True,
    max_num=-1,
    data_name=f'{name}@{seed}',
    seed=seed,
    force_save=True,
    tag=f'CVGAE-{name}@{seed}',
    use_tqdm=False,
)
