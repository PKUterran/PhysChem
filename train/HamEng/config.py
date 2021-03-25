FITTER_CONFIG = {
    'LR': 1e-5,
    'GAMMA': 0.95,
    'DECAY': 4e-5,
    'DROPOUT': 0.2,
    'EPOCH': 30,
    'BATCH': 32,
    'PACK': 1,

    'HGN_LAYERS': 20,
    'TAU': 0.025,
    'PQ_DIM': 32,
    'GAMMA_S': 0.000,
    'GAMMA_C': 0.000,
    'GAMMA_A': 0.001,
    'DISSIPATE': True,
    'LSTM': True,
    'DISTURB': False,
}

FITTER_CONFIG_QM9 = FITTER_CONFIG.copy()
FITTER_CONFIG_QM9.update({
    'MAX_DICT': 4096,
})
