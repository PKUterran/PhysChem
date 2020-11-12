DEFAULT_CONFIG = {
    # model
    'HV_DIM': 64,
    'HE_DIM': 64,
    'HM_DIM': 64,
    'MV_DIM': 64,
    'ME_DIM': 64,
    'MM_DIM': 64,
    'PQ_DIM': 16,
    'N_LAYER': 4,
    'N_HOP': 2,
    'N_ITERATION': 10,
    'N_GLOBAL': 2,
    'MESSAGE_TYPE': 'naive',
    'UNION_TYPE': 'gru',
    'TAU': 0.02,
    'DROPOUT': 0.2,

    'EPOCH': 50,
    'BATCH': 32,
    'LAMBDA': 1,
    'LR': 1e-4,
    'DECAY': 1e-5,
}

QM9_CONFIG = DEFAULT_CONFIG.copy()
QM9_CONFIG.update({

})
