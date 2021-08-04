import numpy as np
import tqdm
import torch
from typing import List, Dict, Tuple, Union, Any
from rdkit import Chem
from rdkit.Chem.rdchem import Mol as Molecule
from rdkit.Chem import AllChem
from rdkit.Chem.AllChem import AlignMol, CalcRMS

from data.encode import encode_mols_generator, num_atom_features, num_bond_features
from train.utils.cache_batch import get_mol_positions
from train.utils.rdkit import rdkit_mol_positions
from .rebuild import rebuild_qm9, rebuild_cvgae, rebuild_hameng
from .vis_derive import generate_derive


def compare_conf(smiles: str, source_conf: np.ndarray, target_conf: np.ndarray) -> float:
    source_mol, target_mol = Chem.MolFromSmiles(smiles), Chem.MolFromSmiles(smiles)
    AllChem.EmbedMolecule(source_mol)
    AllChem.EmbedMolecule(target_mol)
    # print(smiles)
    # print(source_conf)
    # print(target_conf)
    for i, pos in enumerate(source_conf):
        source_mol.GetConformer().SetAtomPosition(i, [float(pos[0]), float(pos[1]), float(pos[2])])
    for i, pos in enumerate(target_conf):
        target_mol.GetConformer().SetAtomPosition(i, [float(pos[0]), float(pos[1]), float(pos[2])])
    # print(source_mol.GetConformer().GetPositions())
    # print(target_mol.GetConformer().GetPositions())
    rms = AlignMol(source_mol, target_mol)
    # print(rms)
    return rms


def eval_rmsd_with_mols(list_mols: List[Molecule], tag: str, special_config: dict, use_cuda=False):
    np.set_printoptions(suppress=True, precision=3, linewidth=200)
    atom_dim, bond_dim = num_atom_features(), num_bond_features()
    model, classifier = rebuild_qm9(atom_dim, bond_dim, tag, special_config, use_cuda)
    cvgae_model, conf_gen_c = rebuild_cvgae(atom_dim, bond_dim, use_cuda=use_cuda)
    hameng_model, conf_gen_h = rebuild_hameng(atom_dim, bond_dim, use_cuda=use_cuda)
    t = tqdm.tqdm(enumerate(encode_mols_generator(list_mols)), total=len(list_mols))

    guess_ls = []
    rdkit_ls = []
    cvgae_ls = []
    hameng_ls = []
    geomnn_ls = []
    better = 0
    for idx, mol_info in t:
        if len(list_mols[idx].GetAtoms()) == 1:
            continue
        try:
            smiles = Chem.MolToSmiles(list_mols[idx])
            # real
            real_conf = get_mol_positions(list_mols[idx])

            # guess
            guess_conf = np.zeros(shape=[len(list_mols[idx].GetAtoms()), 3])
            guess_l = compare_conf(smiles, guess_conf, real_conf)

            # rdkit
            rdkit_conf = rdkit_mol_positions(list_mols[idx])
            rdkit_l = compare_conf(smiles, rdkit_conf, real_conf)

            # CVGAE
            _, list_q = generate_derive(cvgae_model, mol_info, conf_gen_c)
            cvgae_conf = list_q[0]
            cvgae_l = compare_conf(smiles, cvgae_conf, real_conf)

            # HamEng
            list_p, list_q = generate_derive(hameng_model, mol_info, conf_gen_h)
            hameng_conf = list_q[-1]
            hameng_l = compare_conf(smiles, hameng_conf, real_conf)

            # GeomNN
            list_p, list_q = generate_derive(model, mol_info)
            geomnn_conf = list_q[-1]
            geomnn_l = compare_conf(smiles, geomnn_conf, real_conf)

            guess_ls.append(guess_l)
            rdkit_ls.append(rdkit_l)
            cvgae_ls.append(cvgae_l)
            hameng_ls.append(hameng_l)
            geomnn_ls.append(geomnn_l)
            if geomnn_l <= rdkit_l:
                better += 1
        except ValueError:
            pass

        if idx % 1000 == 999:
            print(f'Available: {len(rdkit_ls)}/{idx + 1}')
            print(f'Guess: {np.mean(guess_ls)}')
            print(f'RDKit: {np.mean(rdkit_ls)}')
            print(f'CVGAE: {np.mean(cvgae_ls)}')
            print(f'HamEng: {np.mean(hameng_ls)}')
            print(f'GeomNN: {np.mean(geomnn_ls)}')
            print(f'GeomNN better than RDKit: {better}/{len(rdkit_ls)}')

    print(f'Available: {len(rdkit_ls)}/{len(list_mols)}')
    print(f'Guess: {np.mean(guess_ls)}')
    print(f'RDKit: {np.mean(rdkit_ls)}')
    print(f'CVGAE: {np.mean(cvgae_ls)}')
    print(f'HamEng: {np.mean(hameng_ls)}')
    print(f'GeomNN: {np.mean(geomnn_ls)}')
    print(f'GeomNN better than RDKit: {better}/{len(rdkit_ls)}')
