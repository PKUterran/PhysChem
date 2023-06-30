from net.config import ConfType
from train.CVGAE.train_qm9 import train_qm9, QMDataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pos', type=int, default=0)
arg = parser.parse_args()
pos = arg.pos

if pos == 0:
    conf_type = ConfType.NONE
    name = 'QM9-Xconf'
else:
    conf_type = ConfType.RDKIT
    name = 'QM9-rdkit'

train_qm9(
    special_config={
        'HV_DIM': 128,
        'HE_DIM': 64,

        'EPOCH': 300,
        'BATCH': 20,
        'SAMPLE': 5,
        'LAMBDA': 1,
        'LR': 2e-6,
        'GAMMA': 0.995,
        'DECAY': 1e-5,

        'CONF_TYPE': conf_type,
    },
    dataset=QMDataset.QM9,
    use_cuda=True,
    max_num=-1,
    data_name=f'CVGAE-{name}',
    seed=0,
    force_save=False,
    tag=f'CVGAE-{name}',
    use_tqdm=False,
)
