from net.config import ConfType
from train.train_qm9 import train_qm9, QMDataset

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

        'EPOCH': 300,
        'BATCH': 64,
        'PACK': 1,
        'CONF_LOSS': 'H_ADJ3',
        'LAMBDA': 100,
        'LR': 1e-4,
        'GAMMA': 0.995,
        'DECAY': 1e-5,

        'CONF_TYPE': ConfType.NEWTON,
    },
    dataset=QMDataset.QM7,
    use_cuda=True,
    max_num=-1,
    data_name='QM7',
    seed=0,
    force_save=True,
    tag='QM7',
    use_tqdm=False,
)
