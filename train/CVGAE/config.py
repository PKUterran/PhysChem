from net.config import ConfType

DEFAULT_CONFIG = {
    # model
    'HV_DIM': 128,
    'HE_DIM': 64,

    'EPOCH': 300,
    'BATCH': 20,
    'SAMPLE': 5,
    'LAMBDA': 1,
    'LR': 2e-6,
    'GAMMA': 0.995,
    'DECAY': 1e-5,

    'CONF_TYPE': ConfType.NEWTON,
}

QM9_CONFIG = DEFAULT_CONFIG.copy()
QM9_CONFIG.update({
})
