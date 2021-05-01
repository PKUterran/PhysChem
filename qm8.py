from net.config import ConfType
from train.train_qm9 import train_qm9, QMDataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=16880611)
parser.add_argument('--pos', type=int, default=0)
arg = parser.parse_args()
seed = arg.seed
pos = arg.pos

if pos == 0:
    conf_type = ConfType.NONE
    name = 'QM8-Xconf'
elif pos == 1:
    conf_type = ConfType.RDKIT
    name = 'QM8-rdkit'
elif pos == 2:
    conf_type = ConfType.NEWTON
    name = 'QM8'
elif pos == 3:
    conf_type = ConfType.ONLY
    name = 'QM8-Oconf'
elif pos == 4:
    conf_type = ConfType.REAL
    name = 'QM8-real'
else:
    conf_type = ConfType.SINGLE_CHANNEL
    name = 'QM8-single'

train_qm9(
    special_config={
        'CLASSIFIER_HIDDENS': [],

        'INIT_GCN_H_DIMS': [128],
        'INIT_GCN_O_DIM': 128,
        'INIT_LSTM_LAYERS': 2,
        'INIT_LSTM_O_DIM': 128,

        'HV_DIM': 200,
        'HE_DIM': 100,
        'HM_DIM': 300,
        'MV_DIM': 200,
        'ME_DIM': 100,
        'MM_DIM': 300,
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
        'DROPOUT': 0.5,

        'EPOCH': 100,
        'BATCH': 9,
        'PACK': 9,
        'CONF_LOSS': 'H_ADJ3',
        'LAMBDA': 10,
        'LR': 1e-5,
        'GAMMA': 0.95,
        'DECAY': 1e-3,

        'CONF_TYPE': conf_type,
    },
    dataset=QMDataset.QM8,
    use_cuda=True,
    max_num=-1,
    data_name=f'{name}@{seed}',
    seed=seed,
    force_save=True,
    tag=f'{name}@{seed}',
    use_tqdm=False,
)
