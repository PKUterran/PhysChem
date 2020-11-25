DEFAULT_CONFIG = {
    # model
    'HV_DIM': 64,
    'HE_DIM': 64,
    'HM_DIM': 64,
    'MV_DIM': 64,
    'ME_DIM': 64,
    'MM_DIM': 64,
    'PQ_DIM': 8,
    'N_LAYER': 4,
    'N_HOP': 2,
    'N_ITERATION': 10,
    'N_GLOBAL': 2,
    'MESSAGE_TYPE': 'triplet',
    'UNION_TYPE': 'gru',
    'DERIVATION_TYPE': 'newton',
    'TAU': 0.02,
    'DROPOUT': 0.0,

    'EPOCH': 50,
    'BATCH': 20,
    'LAMBDA': 1,
    'LR': 1e-4,
    'DECAY': 2e-5,
}

QM9_CONFIG = DEFAULT_CONFIG.copy()
QM9_CONFIG.update({

})
