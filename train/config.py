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
    'N_ITERATION': 4,
    'N_GLOBAL': 3,
    'MESSAGE_TYPE': 'triplet',
    'UNION_TYPE': 'gru',
    'DERIVATION_TYPE': 'newton',
    'TAU': 0.05,
    'DROPOUT': 0.0,

    'EPOCH': 200,
    'BATCH': 20,
    'LAMBDA': 1,
    'LR': 1e-5,
    'GAMMA': 1.00,
    'DECAY': 1e-5,

    'CONF_TYPE': ConfType.NEWTON,
}

QM9_CONFIG = DEFAULT_CONFIG.copy()
QM9_CONFIG.update({

})
