DEFAULT_CONFIG = {
    # model
    'HV_DIM': 128,
    'HE_DIM': 128,
    'HM_DIM': 128,
    'MV_DIM': 128,
    'ME_DIM': 128,
    'MM_DIM': 128,
    'PQ_DIM': 16,
    'N_LAYER': 4,
    'N_ITERATION': 5,
    'N_GLOBAL': 3,
    'MESSAGE_TYPE': 'naive',
    'UNION_TYPE': 'gru',
    'TAU': 0.1,
    'DROPOUT': 0.0,

    'EPOCH': 50,
    'BATCH': 32,
    'LAMBDA': 1,
    'LR': 1e-2,
    'DECAY': 1e-5,
}

QM9_CONFIG = DEFAULT_CONFIG.copy()
QM9_CONFIG.update({

})
