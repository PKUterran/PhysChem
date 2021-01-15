from net.config import ConfType


DEFAULT_CONFIG = {
    # model
    'CLASSIFIER_HIDDENS': [],
    'HV_DIM': 64,
    'HE_DIM': 32,
    'HM_DIM': 256,
    'MV_DIM': 64,
    'ME_DIM': 32,
    'MM_DIM': 256,
    'PQ_DIM': 3,
    'N_LAYER': 2,
    'N_HOP': 1,
    'N_ITERATION': 5,
    'N_GLOBAL': 2,
    'MESSAGE_TYPE': 'triplet',
    'UNION_TYPE': 'gru',
    'GLOBAL_TYPE': 'inductive',
    'DERIVATION_TYPE': 'newton',
    'TAU': 0.05,
    'DROPOUT': 0.0,

    'EPOCH': 300,
    'BATCH': 20,
    'PACK': 1,
    'LAMBDA': 1,
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
    'N_GLOBAL': 4,
    'DROPOUT': 0.2,

    'EPOCH': 250,
    'BATCH': 2,
    'PACK': 16,
    'LR': 1e-4,
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

    'N_LAYER': 1,
    'N_ITERATION': 1,
    'N_HOP': 2,
    'N_GLOBAL': 2,
    'DROPOUT': 0.5,

    'EPOCH': 400,
    'BATCH': 2,
    'PACK': 64,
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
    'N_ITERATION': 1,
    'N_HOP': 1,
    'N_GLOBAL': 2,
    'DROPOUT': 0.2,

    'EPOCH': 400,
    'BATCH': 2,
    'PACK': 128,
    'LR': 3e-3,
    'GAMMA': 0.995,
    'DECAY': 1e-5,

    'CONF_TYPE': ConfType.RDKIT,
})

FREESOLV_CONFIG = DEFAULT_CONFIG.copy()
FREESOLV_CONFIG.update({
    'CLASSIFIER_HIDDENS': [],
    'HV_DIM': 160,
    'HE_DIM': 160,
    'HM_DIM': 160,
    'MV_DIM': 160,
    'ME_DIM': 160,
    'MM_DIM': 160,

    'N_LAYER': 1,
    'N_ITERATION': 1,
    'N_HOP': 2,
    'N_GLOBAL': 2,
    'DROPOUT': 0.2,

    'EPOCH': 400,
    'BATCH': 2,
    'PACK': 128,
    'LR': 3e-3,
    'GAMMA': 0.995,
    'DECAY': 1e-5,

    'CONF_TYPE': ConfType.RDKIT,
})

