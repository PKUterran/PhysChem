DEFAULT_CONFIG = {
    # model
    'HV_DIM': 64,
    'HE_DIM': 64,
    'HM_DIM': 64,
    'MV_DIM': 64,
    'ME_DIM': 64,
    'MM_DIM': 64,
    'PQ_DIM': 3,
    'N_LAYER': 3,
    'N_HOP': 2,
    'N_ITERATION': 16,
    'N_GLOBAL': 2,
    'MESSAGE_TYPE': 'triplet',
    'UNION_TYPE': 'gru',
    'DERIVATION_TYPE': 'newton',
    'TAU': 0.01,
    'DROPOUT': 0.0,

    'EPOCH': 200,
    'BATCH': 20,
    'LAMBDA': 1,
    'LR': 1e-4,
    'DECAY': 1e-5,

    'Xconf': False,
}

QM9_CONFIG = DEFAULT_CONFIG.copy()
QM9_CONFIG.update({

})
