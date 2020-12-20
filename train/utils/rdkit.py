import numpy as np
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol as Molecule


def rdkit_mol_positions(mol: Molecule):
    position = np.zeros([len(mol.GetAtoms()), 3], np.float)
    try:
        AllChem.EmbedMolecule(mol)
        conf = mol.GetConformer()
        position = conf.GetPositions()
    except ValueError:
        pass
    return position


if __name__ == '__main__':
    p = r'D:\geom_data\rdkit_folder\qm9\C#C.pickle'
    import pickle
    with open(p, 'rb') as fp:
        m = pickle.load(fp)
    confs = m.GetConformers()
    print(len(confs))
    print(confs[0].GetPositions())
    print(rdkit_mol_positions(m))
    confs = m.GetConformers()
    print(len(confs))
