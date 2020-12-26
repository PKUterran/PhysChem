from net.config import ConfType


DEFAULT_CONFIG = {
    # model
    'CLASSIFIER_HIDDENS': [128],
    'HV_DIM': 128,
    'HE_DIM': 128,
    'HM_DIM': 128,
    'MV_DIM': 128,
    'ME_DIM': 128,
    'MM_DIM': 128,
    'PQ_DIM': 3,
    'N_LAYER': 1,
    'N_HOP': 1,
    'N_ITERATION': 5,
    'N_GLOBAL': 3,
    'MESSAGE_TYPE': 'triplet',
    'UNION_TYPE': 'gru',
    'DERIVATION_TYPE': 'newton',
    'TAU': 0.05,
    'DROPOUT': 0.0,

    'EPOCH': 300,
    'BATCH': 20,
    'LAMBDA': 1,
    'LR': 2e-6,
    'GAMMA': 0.995,
    'DECAY': 1e-5,

    'CONF_TYPE': ConfType.NEWTON,
}

QM9_CONFIG = DEFAULT_CONFIG.copy()
QM9_CONFIG.update({

})

LIPOP_CONFIG = DEFAULT_CONFIG.copy()
LIPOP_CONFIG.update({
    'CLASSIFIER_HIDDENS': [128],
    'HV_DIM': 256,
    'HE_DIM': 256,
    'HM_DIM': 256,
    'MV_DIM': 256,
    'ME_DIM': 256,
    'MM_DIM': 256,

    'N_ITERATION': 2,
    'N_GLOBAL': 4,

    'BATCH': 5,
    'LR': 1e-5,
    'GAMMA': 0.99,
    'DECAY': 1e-5,

    'CONF_TYPE': ConfType.RDKIT,
})

TOX21_CONFIG = DEFAULT_CONFIG.copy()
TOX21_CONFIG.update({
    'CLASSIFIER_HIDDENS': [128],
    'HV_DIM': 256,
    'HE_DIM': 256,
    'HM_DIM': 256,
    'MV_DIM': 256,
    'ME_DIM': 256,
    'MM_DIM': 256,

    'N_ITERATION': 3,
    'N_GLOBAL': 3,

    'BATCH': 5,
    'LR': 2e-6,
    'GAMMA': 0.995,
    'DECAY': 1e-4,

    'CONF_TYPE': ConfType.RDKIT,
})

ESOL_CONFIG = DEFAULT_CONFIG.copy()
ESOL_CONFIG.update({
    'CLASSIFIER_HIDDENS': [128],
    'HV_DIM': 256,
    'HE_DIM': 256,
    'HM_DIM': 256,
    'MV_DIM': 256,
    'ME_DIM': 256,
    'MM_DIM': 256,

    'N_ITERATION': 3,
    'N_GLOBAL': 3,

    'BATCH': 5,
    'LR': 1e-5,
    'GAMMA': 0.995,
    'DECAY': 1e-4,

    'CONF_TYPE': ConfType.RDKIT,
})

FREESOLV_CONFIG = DEFAULT_CONFIG.copy()
FREESOLV_CONFIG.update({
    'CLASSIFIER_HIDDENS': [128],
    'HV_DIM': 256,
    'HE_DIM': 256,
    'HM_DIM': 256,
    'MV_DIM': 256,
    'ME_DIM': 256,
    'MM_DIM': 256,

    'N_ITERATION': 3,
    'N_GLOBAL': 3,

    'BATCH': 5,
    'LR': 1e-5,
    'GAMMA': 0.995,
    'DECAY': 5e-4,

    'CONF_TYPE': ConfType.RDKIT,
})

