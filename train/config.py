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

    'N_ITERATION': 1,
    'N_HOP': 2,
    'N_GLOBAL': 2,

    'EPOCH': 400,
    'BATCH': 4,
    'PACK': 8,
    'LR': 2e-5,
    'GAMMA': 0.995,
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
    'CLASSIFIER_HIDDENS': [],
    'HV_DIM': 160,
    'HE_DIM': 160,
    'HM_DIM': 160,
    'MV_DIM': 160,
    'ME_DIM': 160,
    'MM_DIM': 160,

    'N_ITERATION': 1,
    'N_HOP': 2,
    'N_GLOBAL': 2,
    'DROPOUT': 0.2,

    'EPOCH': 200,
    'BATCH': 4,
    'PACK': 32,
    'LR': 3e-3,
    'GAMMA': 0.99,
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

    'N_ITERATION': 1,
    'N_HOP': 2,
    'N_GLOBAL': 2,
    'DROPOUT': 0.2,

    'EPOCH': 400,
    'BATCH': 4,
    'PACK': 20,
    'LR': 3e-3,
    'GAMMA': 0.995,
    'DECAY': 1e-5,

    'CONF_TYPE': ConfType.RDKIT,
})

