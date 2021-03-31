from net.config import ConfType

DEFAULT_CONFIG = {
    # model
    'CLASSIFIER_HIDDENS': [],

    # 'INIT_GCN_H_DIMS': [],
    # 'INIT_GCN_O_DIM': 32,
    # 'INIT_LSTM_LAYERS': 1,
    # 'INIT_LSTM_O_DIM': 0,

    'INIT_GCN_H_DIMS': [128],
    'INIT_GCN_O_DIM': 128,
    'INIT_LSTM_LAYERS': 2,
    'INIT_LSTM_O_DIM': 128,

    'HV_DIM': 128,
    'HE_DIM': 64,
    'HM_DIM': 256,
    'MV_DIM': 128,
    'ME_DIM': 64,
    'MM_DIM': 256,
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
    'BATCH': 20,
    'PACK': 1,
    'CONF_LOSS': 'H_ADJ3',
    'LAMBDA': 100,
    'LR': 2e-6,
    'GAMMA': 0.995,
    'DECAY': 1e-5,

    'CONF_TYPE': ConfType.NEWTON,
}

QM9_CONFIG = DEFAULT_CONFIG.copy()
QM9_CONFIG.update({
    'DROPOUT': 0.5,
})

LIPOP_CONFIG = DEFAULT_CONFIG.copy()
LIPOP_CONFIG.update({
    'CLASSIFIER_HIDDENS': [],
    'HV_DIM': 256,
    'HE_DIM': 256,
    'HM_DIM': 256,
    'MV_DIM': 256,
    'ME_DIM': 256,
    'MM_DIM': 256,

    'N_LAYER': 2,
    'N_ITERATION': 1,
    'N_HOP': 1,
    'N_GLOBAL': 2,
    'DROPOUT': 0.2,

    'EPOCH': 400,
    'BATCH': 1,
    'PACK': 25,
    'LAMBDA': 1,
    'LR': 1e-4,
    'GAMMA': 0.995,
    'DECAY': 1e-5,

    'CONF_TYPE': ConfType.RDKIT,
})

ESOL_CONFIG = DEFAULT_CONFIG.copy()
ESOL_CONFIG.update({
    'CLASSIFIER_HIDDENS': [],
    'HV_DIM': 160,
    'HE_DIM': 160,
    'HM_DIM': 160,
    'MV_DIM': 160,
    'ME_DIM': 160,
    'MM_DIM': 160,

    'N_LAYER': 2,
    'N_ITERATION': 4,
    'N_HOP': 1,
    'N_GLOBAL': 2,
    'DROPOUT': 0.2,

    'EPOCH': 400,
    'BATCH': 1,
    'PACK': 160,
    'LAMBDA': 1,
    'LR': 3e-3,
    'GAMMA': 0.995,
    'DECAY': 1e-5,

    'CONF_TYPE': ConfType.RDKIT,
})

FREESOLV_CONFIG = DEFAULT_CONFIG.copy()
FREESOLV_CONFIG.update({
    'CLASSIFIER_HIDDENS': [],
    'HV_DIM': 120,
    'HE_DIM': 120,
    'HM_DIM': 120,
    'MV_DIM': 120,
    'ME_DIM': 120,
    'MM_DIM': 120,

    'N_LAYER': 2,
    'N_ITERATION': 4,
    'N_HOP': 1,
    'N_GLOBAL': 2,
    'DROPOUT': 0.2,

    'EPOCH': 400,
    'BATCH': 1,
    'PACK': 160,
    'LAMBDA': 1,
    'LR': 3e-3,
    'GAMMA': 0.995,
    'DECAY': 1e-5,

    'CONF_TYPE': ConfType.RDKIT,
})

TOX21_CONFIG = DEFAULT_CONFIG.copy()
TOX21_CONFIG.update({
    'CLASSIFIER_HIDDENS': [],
    'HV_DIM': 200,
    'HE_DIM': 200,
    'HM_DIM': 200,
    'MV_DIM': 200,
    'ME_DIM': 200,
    'MM_DIM': 200,

    'N_LAYER': 2,
    'N_ITERATION': 4,
    'N_HOP': 1,
    'N_GLOBAL': 2,
    'DROPOUT': 0.5,

    'EPOCH': 200,
    'BATCH': 2,
    'PACK': 64,
    'LAMBDA': 1,
    'LR': 1e-4,
    'GAMMA': 0.99,
    'DECAY': 1e-5,

    'CONF_TYPE': ConfType.RDKIT,
})
