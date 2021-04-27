import os
import json
import rdkit.Chem as Chem
from rdkit.Chem.AllChem import EmbedMolecule
from typing import List


def convert_json2pdb(input_dir: str, output_dir: str, names: List = None):
    files = os.listdir(input_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for f in files:
        if not f.endswith('.json'):
            continue
        if names is not None and f[:-5] not in names:
            continue

        with open(f'{input_dir}/{f}') as fp:
            dic = json.load(fp)
        tag = dic['tag']
        smiles = dic['smiles']
        pos = dic['pos']
        mol = Chem.MolFromSmiles(smiles)
        EmbedMolecule(mol)
        for i in range(len(mol.GetAtoms())):
            mol.GetConformer().SetAtomPosition(i, pos[i])
        Chem.MolToPDBFile(mol, filename=f'{output_dir}/{tag}.pdb')


if __name__ == '__main__':
    convert_json2pdb('json_smiles', 'pdb_smiles')
