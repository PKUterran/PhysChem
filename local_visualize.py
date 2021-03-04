import pandas as pd

from net.config import ConfType
from visualize.vis_alignment import vis_alignment
from visualize.vis_bond import vis_bond
from visualize.vis_derive import vis_derive

if __name__ == '__main__':
    vis_derive(
        list_smiles=[
            'c1ccccc1C#N',
            'c1ccccc1C(O)=O',
            'c1cn(cn1)C2CC2',
            'NC(C)C(O)=O',
            'CCCCCC',
            'C1CCCCC1',
            'c1ccccc1',
            'Cc1cccc(c1C)Nc2ccccc2C(=O)O',
        ],
        tag='QM9',
        special_config={
            # 'INIT_GCN_H_DIMS': [128],
            # 'INIT_GCN_O_DIM': 128,
            # 'INIT_LSTM_LAYERS': 4,
            # 'INIT_LSTM_O_DIM': 128,
            # 'HV_DIM': 128,
            # 'HE_DIM': 128,
            # 'HM_DIM': 128,
            # 'MV_DIM': 128,
            # 'ME_DIM': 128,
            # 'MM_DIM': 128,
            # 'PQ_DIM': 3,
            # 'N_LAYER': 2,
            # 'N_HOP': 1,
            # 'N_ITERATION': 4,
            # 'N_GLOBAL': 2,
            # 'TAU': 0.05,
            # 'DISSA': 1.0,
            # 'EPOCH': 200,
            # 'BATCH': 32,
            # 'LAMBDA': 100,
            # 'LR': 1e-4,
            # 'GAMMA': 0.99,
            # 'DECAY': 1e-5,

            # 'CONF_LOSS': 'DL',
            # 'LAMBDA': 1,
            # 'MESSAGE_TYPE': 'triplet-mean',
        },
        # tag='QM9',
        # special_config={
        #     # 'CONF_LOSS': 'DL',
        #     # 'LAMBDA': 1,
        #     # 'MESSAGE_TYPE': 'triplet-mean',
        # },
    )
    # list_smiles = pd.read_csv('data/qm9/qm9.csv').values[3:40003, 1]
    # vis_bond(
    #     list_smiles=list_smiles,
    #     tag='QM9-M',
    #     special_config={
    #
    #     },
    # )
