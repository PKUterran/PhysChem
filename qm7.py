from net.config import ConfType
from train.train_qm9 import train_qm9, QMDataset

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
elif pos == 1:
    conf_type = ConfType.RDKIT
    name = 'QM7-rdkit'
else:
    conf_type = ConfType.NEWTON
    name = 'QM7'

train_qm9(
    special_config={
        'CLASSIFIER_HIDDENS': [],

        'INIT_GCN_H_DIMS': [32],
        'INIT_GCN_O_DIM': 32,
        'INIT_LSTM_LAYERS': 2,
        'INIT_LSTM_O_DIM': 32,

        'HV_DIM': 32,
        'HE_DIM': 16,
        'HM_DIM': 64,
        'MV_DIM': 32,
        'ME_DIM': 16,
        'MM_DIM': 64,
        'PQ_DIM': 3,
        'N_LAYER': 2,
        'N_HOP': 1,
        'N_ITERATION': 4,
        'N_GLOBAL': 2,
        'MESSAGE_TYPE': 'triplet',
        'UNION_TYPE': 'gru',
        'GLOBAL_TYPE': 'inductive',
        'DERIVATION_TYPE': 'newton',
        'TAU': 0.25,
        'DISSA': 1.0,
        'DROPOUT': 0.0,

        'EPOCH': 200,
        'BATCH': 64,
        'PACK': 1,
        'CONF_LOSS': 'H_ADJ3',
        'LAMBDA': 100,
        'LR': 2e-5,
        'GAMMA': 0.99,
        'DECAY': 1e-5,

        'CONF_TYPE': ConfType.NEWTON,
    },
    dataset=QMDataset.QM7,
    use_cuda=True,
    max_num=-1,
    data_name=f'{name}@{seed}',
    seed=0,
    force_save=True,
    tag=f'{name}@{seed}',
    use_tqdm=False,
)
