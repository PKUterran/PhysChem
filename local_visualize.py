
from net.config import ConfType
from visualize.vis_alignment import vis_alignment

if __name__ == '__main__':
    vis_alignment(
        list_smiles=[
            'c1ccccc1C#N',
            'c1ccccc1C(O)=O',
            'c1cn(cn1)C2CC2',
            'NC(C)C(O)=O',
        ],
        tag='QM9',
        special_config={

        },
    )
