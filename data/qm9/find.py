import tqdm
from rdkit import Chem, DataStructs


def find(smiles: str) -> int:
    supplier = Chem.SDMolSupplier('gdb9.sdf')
    target_rdk = Chem.RDKFingerprint(Chem.MolFromSmiles(smiles))
    cnt = 0
    for m in tqdm.tqdm(supplier):
        if m is None or not m.GetProp("_Name").startswith("gdb"):
            continue
        rdk = Chem.RDKFingerprint(m)
        if DataStructs.FingerprintSimilarity(rdk, target_rdk) > 1 - 1e-3:
            print(Chem.MolToSmiles(m))
            print(cnt)
            return int(m.GetProp("_Name")[4:])
        cnt += 1
    return -1


if __name__ == '__main__':
    print(find('CCCCC'))


'''
NC(C)C(O)=O 285 286
C([C@@H](C(=O)O)N)C(=O)O -1
N[C@H](C(O)=O)CC(N)=O 61158 61439
c1c(C=CN2)c2ccc1 24408 24492
c1ccccc1C#N 5336 5347
CC(C)CCCCCC 122508 123139
CCCCC 132 133
'''
