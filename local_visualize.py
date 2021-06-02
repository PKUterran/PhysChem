import pandas as pd
from rdkit import Chem

from data.qm9.load_qm9 import load_qm9
from net.config import ConfType
from visualize.vis_alignment import vis_alignment
from visualize.vis_bond import vis_bond
from visualize.vis_derive import vis_derive_with_mols, vis_derive_with_smiles


def vis_derive_script():
    # mols, _ = load_qm9()
    # print(len(mols))
    # print(Chem.MolToSmiles(mols[61440]))
    # print(Chem.MolToSmiles(mols[24408]))
    # exit()
    # vis_derive_with_mols(
    #     list_mols=[
    #         mols[5336],
    #         mols[132],
    #     ],
    #     tag='QM9',
    #     special_config={
    #
    #     },)
    # vis_derive_with_smiles(
    #     list_smiles=[
    #         'O=Cc1cc(C#N)ccc1',
    #         'C(C(C(=O)O)N)C(=O)O',
    #         'CC(=O)OC1=CC=CC=C1C(=O)O',
    #     ],
    #     tag='QM9',
    #     special_config={
    #
    #     },
    # )
    vis_alignment(
        list_smiles=[
            'c1ccccc1C#N',
            'OCC(O)=O',
        ],
        tag='QM9',
        special_config={

        },
    )


if __name__ == '__main__':
    vis_derive_script()
    # list_smiles = pd.read_csv('data/geom_qm9/geom_qm9.csv').values[3:40003, 1]
    # vis_bond(
    #     list_smiles=list_smiles,
    #     tag='QM9-M',
    #     special_config={
    #
    #     },
    # )
    pass
