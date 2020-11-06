import json
import time
import pickle
import rdkit
import pandas as pd
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdmolops import RemoveHs, RemoveAllHs
from rdkit.Chem.AllChem import GetMorganFingerprint, GetHashedMorganFingerprint, GetMorganFingerprintAsBitVect

from data.config import *


def compare(f1, f2):
    s = DataStructs.FingerprintSimilarity(f1, f2)
    return s > 1 - 1e-5


def qm9_dict(start=0):
    fp = open(QM9_RDKIT_SUMMARY_PATH)
    d: dict = json.load(fp)
    fp.close()
    k = list(d.keys())
    k_len = len(k)
    print(k_len)
    # path = f'{RDKIT_FOLDER_DIR}/{d["C[C@H]1[C@H]2O[C@]1(C)[C@]2(C)O"]["pickle_path"]}'
    # fp = open(path, 'rb')
    # p: dict = pickle.load(fp)
    # fp.close()
    # confs: list = p['conformers']
    # for conf in confs:
    #     mol = conf['rd_mol']
    #     mol = RemoveAllHs(mol)
    #     ss = [a.GetSymbol() for a in mol.GetAtoms()]
    #     cs = mol.GetConformers()
    #     c = cs[0]
    #     positions = c.GetPositions()
    #     # positions = [list(p) for s, p in zip(ss, positions) if s != 'H']
    #     # ss = [s for s in ss if s != 'H']
    #     print(positions)
    #     print(ss)

    csv = pd.read_csv('../data/qm9/qm9.csv')
    nd = csv.values
    smiles = nd[:, 1]
    all_smiles_len = len(smiles)
    print(all_smiles_len)
    smiles = smiles[start:]
    total = 0
    hit = 0

    f_dir = {}
    csv2json = {}
    for s in smiles:
        if s in k:
            # print(s)
            csv2json[s] = s
            hit += 1
        else:
            flag = 0
            f1 = GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2048)
            for s2 in k:
                lll = len(f_dir.keys())
                if lll % 1000 == 0:
                    print('cached:{:.2f}%'.format(100 * lll / k_len))
                if s2 in f_dir.keys():
                    f2 = f_dir[s2]
                else:
                    try:
                        f2 = GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s2), 2048)
                    except:
                        f2 = None
                    f_dir[s2] = f2
                if not f2:
                    continue
                if compare(f1, f2):
                    hit += 1
                    # print(s, s2)
                    csv2json[s] = s2
                    flag = 1
                    break
            if not flag:
                # print(s, 'None')
                csv2json[s] = ''

        total += 1
        # assert (total - 1) % 2000 == len(csv2json.items()) - 1
        if total % 100 == 0:
            print('processed:{:.2f}%'.format(100 * total / len(smiles)))
        if total % 2000 == 0:
            print('saving...')
            with open('../data/qm9/mid/csv2json_{}-{}.json'.format(
                    start + total - 1999, start + total), 'w+', encoding='utf-8') as fp:
                json.dump(csv2json, fp)
            csv2json.clear()

    with open('../data/qm9/mid/csv2json_tail.json', 'w+', encoding='utf-8') as fp:
        json.dump(csv2json, fp)
    print('hit rate: {:.1f}%'.format(hit * 100 / total))


qm9_dict(124957)
