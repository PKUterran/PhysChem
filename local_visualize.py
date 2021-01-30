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
        ],
        tag='QM9',
        special_config={
            # 'MESSAGE_TYPE': 'triplet-mean',
        },
    )
    # list_smiles = pd.read_csv('data/qm9/qm9.csv').values[3:40003, 1]
    # vis_bond(
    #     list_smiles=list_smiles,
    #     tag='QM9-M',
    #     special_config={
    #
    #     },
    # )
