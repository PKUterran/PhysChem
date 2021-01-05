
from net.config import ConfType
from visualize.vis_alignment import vis_alignment

if __name__ == '__main__':
    vis_alignment(
        list_smiles=['c1ccccc1C#N'],
        tag='QM9-1000-Xconf',
        special_config={
            'CLASSIFIER_HIDDENS': [],
            'HV_DIM': 32,
            'HE_DIM': 32,
            'HM_DIM': 32,
            'MV_DIM': 32,
            'ME_DIM': 32,
            'MM_DIM': 32,

            'N_LAYER': 2,
            'N_ITERATION': 1,
            'N_HOP': 1,
            'N_GLOBAL': 2,
            'DROPOUT': 0.2,

            'EPOCH': 400,
            'BATCH': 2,
            'PACK': 10,
            'LR': 1e-3,
            'GAMMA': 0.995,
            'DECAY': 1e-5,

            'CONF_TYPE': ConfType.NONE,
        },
    )
