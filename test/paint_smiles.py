import os
import rdkit.Chem as Chem
from rdkit.Chem import Draw


def paint_smiles(smiles: str):
    if not os.path.exists('graph'):
        os.mkdir('graph')
    mol = Chem.MolFromSmiles(smiles)
    Draw.MolToFile(mol, f'graph/{smiles}.png')


if __name__ == '__main__':
    paint_smiles('CN(C)C[C@H](C)CN1c3ccccc3Sc2ccccc12')
