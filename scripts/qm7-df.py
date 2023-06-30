from net.config import ConfType
from train.train_qm9 import train_qm9, QMDataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=16880611)
parser.add_argument('--pq', type=int, default=16)
arg = parser.parse_args()
seed = arg.seed
pq = arg.pq
name = f'QM7-df{pq}'
conf_type = ConfType.NEWTON

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
        'PQ_DIM': pq,
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
        'BATCH': 10,
        'PACK': 10,
        'CONF_LOSS': 'H_ADJ3',
        'LAMBDA': 0.1,
        'LR': 1e-4,
        'GAMMA': 0.95,
        'DECAY': 1e-3,

        'CONF_TYPE': conf_type,
    },
    dataset=QMDataset.QM7,
    use_cuda=True,
    max_num=-1,
    data_name=f'{name}@{seed}',
    seed=seed,
    force_save=True,
    tag=f'{name}@{seed}',
    use_tqdm=False,
)
