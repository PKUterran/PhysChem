import os
import time
import torch
import torch.optim as optim
import numpy as np
import pickle

from enum import Enum
from typing import List, Dict
from functools import reduce
from tqdm import tqdm
from rdkit.Chem import MolToSmiles, RemoveAllHs

# from data.geom_qm9.load_qm9 import load_qm9 as load_geom_qm9
from data.qm7.load_qm7 import load_qm7
from data.qm8.load_qm8 import load_qm8
from data.qm9.load_qm9 import load_qm9
from net.baseline.AttentiveFP.AttentiveLayers import Fingerprint
from net.baseline.AttentiveFP.getFeatures import save_smiles_dicts, get_smiles_array
from .config import QM7_CONFIG, QM8_CONFIG
from train.utils.cache_batch import Batch, load_batch_cache, load_encode_mols, batch_cuda_copy
from train.utils.seed import set_seed
from train.utils.loss_functions import multi_mse_loss, multi_mae_loss
from train.utils.save_log import save_log

MODEL_DICT_DIR = 'train/models/AttentiveFP'


class QMDataset(Enum):
    QM7 = 1,
    QM8 = 2,


def train_qm9(special_config: dict = None, dataset=QMDataset.QM7,
              use_cuda=False, max_num=-1, data_name='QM9', seed=0, force_save=False, tag='AttentiveFP-QM9',
              use_tqdm=False):
    # set parameters and seed
    print(f'For {tag}:')
    if dataset == QMDataset.QM7:
        config = QM7_CONFIG.copy()
    else:
        config = QM8_CONFIG.copy()
    if special_config is not None:
        config.update(special_config)
    print('\t CONFIG:')
    for k, v in config.items():
        print(f'\t\t{k}: {v}')
    set_seed(seed, use_cuda=use_cuda)
    np.set_printoptions(suppress=True, precision=3, linewidth=200)

    # load dataset
    print('Loading:')
    # mol_list_weight_mol, mol_properties = load_geom_qm9(max_num)
    # mols = [list_weight_mol[0][1] for list_weight_mol in mol_list_weight_mol]
    if dataset == QMDataset.QM7:
        mols, mol_properties = load_qm7(max_num)
        mols = [RemoveAllHs(mol) for mol in mols]
        not_single_mask = [len(mol.GetAtoms()) > 1 for mol in mols]
        mols = [mols[i] for i in range(len(mols)) if not_single_mask[i]]
        mol_properties = mol_properties[not_single_mask, :]
        mols_info = load_encode_mols(mols, name=data_name, force_save=force_save)
    else:
        mols, mol_properties = load_qm8(max_num)
        mols = [RemoveAllHs(mol) for mol in mols]
        not_single_mask = [len(mol.GetAtoms()) > 1 for mol in mols]
        mols = [mols[i] for i in range(len(mols)) if not_single_mask[i]]
        mol_properties = mol_properties[not_single_mask, :]
        mols_info = load_encode_mols(mols, name=data_name, force_save=force_save)

    smiles_list = [MolToSmiles(m) for m in mols]
    feature_filename = f'train/AttentiveFP/{data_name}.pickle'
    filename = f'train/AttentiveFP/{data_name}'
    if os.path.isfile(feature_filename):
        feature_dicts = pickle.load(open(feature_filename, "rb"))
    else:
        feature_dicts = save_smiles_dicts(smiles_list, filename)

    # normalize properties and cache batches
    mean_p = np.mean(mol_properties, axis=0)
    stddev_p = np.std((mol_properties - mean_p).tolist(), axis=0, ddof=0)
    weights = torch.tensor(stddev_p ** 2, dtype=torch.float32) * 500
    if use_cuda:
        weights = weights.cuda()
    norm_p = (mol_properties - mean_p) / stddev_p
    print(f'\tmean: {mean_p}')
    print(f'\tstd: {stddev_p}')
    # print(f'\tmad: {mad_p}')
    if dataset == QMDataset.QM8:
        print(f'\tweights: {weights.cpu().numpy()}')
    print('Caching Batches...')
    try:
        batch_cache = load_batch_cache(data_name, mols, mols_info, norm_p, batch_size=config['BATCH'],
                                       contains_ground_truth_conf=False, need_mask_matrices=False,
                                       use_cuda=use_cuda, use_tqdm=use_tqdm, force_save=force_save)
    except EOFError:
        batch_cache = load_batch_cache(data_name, mols, mols_info, norm_p, batch_size=config['BATCH'],
                                       contains_ground_truth_conf=False, need_mask_matrices=False,
                                       use_cuda=use_cuda, use_tqdm=use_tqdm, force_save=True)

    # build model
    print('Building Models...')
    model = Fingerprint(
        radius=config['RADIUS'],
        T=config['T'],
        input_feature_dim=39,
        input_bond_dim=10,
        fingerprint_dim=config['DIM'],
        output_units_num=mol_properties.shape[1],
        p_dropout=config['DROPOUT']
    )
    if use_cuda:
        model.cuda()

    # initialize optimization
    parameters = list(model.parameters())
    optimizer = optim.Adam(params=parameters, lr=config['LR'], weight_decay=config['DECAY'])
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=config['GAMMA'])
    print('##### Parameters #####')

    param_size = 0
    for name, param in model.named_parameters():
        print(f'\t\t{name}: {param.shape}')
        param_size += reduce(lambda x, y: x * y, param.shape)
    print(f'\tNumber of parameters: {param_size}')

    # train
    epoch = 0
    logs: List[Dict[str, float]] = []
    best_epoch = 0
    best_metric = 999
    try:
        if not os.path.exists(MODEL_DICT_DIR):
            os.mkdir(MODEL_DICT_DIR)
    except FileExistsError:
        pass

    def train(batches: List[Batch], masks: List[List[int]]):
        model.train()
        optimizer.zero_grad()
        n_batch = len(batches)
        if use_tqdm:
            batches = tqdm(batches, total=n_batch)
        for mask, batch in zip(masks, batches):
            if use_cuda:
                batch = batch_cuda_copy(batch)
            temp_smiles_list = [smiles_list[iii] for iii in mask]
            x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(
                temp_smiles_list, feature_dicts)
            if use_cuda:
                _, pred_p = model.forward(torch.Tensor(x_atom).cuda(), torch.Tensor(x_bonds).cuda(),
                                          torch.cuda.LongTensor(x_atom_index),
                                          torch.cuda.LongTensor(x_bond_index), torch.Tensor(x_mask).cuda())
            else:
                _, pred_p = model.forward(torch.Tensor(x_atom), torch.Tensor(x_bonds),
                                          torch.LongTensor(x_atom_index),
                                          torch.LongTensor(x_bond_index), torch.Tensor(x_mask))
            if dataset == QMDataset.QM8:
                p_losses = multi_mse_loss(pred_p, batch.properties, explicit=True)
                p_loss = sum(p_losses * weights)
            else:
                p_loss = multi_mse_loss(pred_p, batch.properties)
            loss = p_loss
            loss.backward()
            optimizer.step()

    def evaluate(batches: List[Batch], masks: List[List[int]], batch_name: str) -> float:
        model.eval()
        optimizer.zero_grad()
        n_batch = len(batches)
        list_p_loss = []
        list_loss = []
        list_p_multi_mae = []
        list_p_total_mae = []
        if use_tqdm:
            batches = tqdm(batches, total=n_batch)
        for mask, batch in zip(masks, batches):
            if use_cuda:
                batch = batch_cuda_copy(batch)
            temp_smiles_list = [smiles_list[iii] for iii in mask]
            x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(
                temp_smiles_list, feature_dicts)
            if use_cuda:
                _, pred_p = model.forward(torch.Tensor(x_atom).cuda(), torch.Tensor(x_bonds).cuda(),
                                          torch.cuda.LongTensor(x_atom_index),
                                          torch.cuda.LongTensor(x_bond_index), torch.Tensor(x_mask).cuda())
            else:
                _, pred_p = model.forward(torch.Tensor(x_atom), torch.Tensor(x_bonds),
                                          torch.LongTensor(x_atom_index),
                                          torch.LongTensor(x_bond_index), torch.Tensor(x_mask))
            if dataset == QMDataset.QM8:
                p_losses = multi_mse_loss(pred_p, batch.properties, explicit=True)
                p_loss = sum(p_losses * weights)
            else:
                p_loss = multi_mse_loss(pred_p, batch.properties)
            loss = p_loss
            list_p_loss.append(p_loss.cpu().item())
            list_loss.append(loss.cpu().item())

            p_multi_mae = multi_mae_loss(pred_p, batch.properties, explicit=True)
            p_total_mae = p_multi_mae.sum()
            list_p_multi_mae.append(p_multi_mae.cpu().detach().numpy())
            list_p_total_mae.append(p_total_mae.cpu().item())

        print(f'\t\t\tP LOSS: {sum(list_p_loss) / n_batch}')
        print(f'\t\t\tTOTAL LOSS: {sum(list_loss) / n_batch}')
        print(f'\t\t\tPROPERTIES MULTI-MAE: {sum(list_p_multi_mae) * stddev_p / n_batch}')
        if dataset == QMDataset.QM8:
            total_mae = np.sum(sum(list_p_multi_mae) * stddev_p / n_batch)
        else:
            total_mae = sum(list_p_total_mae) / n_batch
        print(f'\t\t\tPROPERTIES TOTAL MAE: {total_mae}')
        logs[-1].update({
            f'{batch_name}_p_loss': sum(list_p_loss) / n_batch,
            f'{batch_name}_loss': sum(list_loss) / n_batch,
            f'{batch_name}_p_metric': total_mae,
            f'{batch_name}_multi_p_metric': list(sum(list_p_multi_mae) * stddev_p / n_batch),
        })
        return sum(list_p_total_mae) / n_batch

    for _ in range(config['EPOCH']):
        epoch += 1
        t0 = time.time()

        logs.append({'epoch': epoch})
        print()
        print(f'##### IN EPOCH {epoch} #####')
        print('\tCurrent LR: {:.3e}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        print('\t\tTraining:')
        train(batch_cache.train_batches, batch_cache.train_masks)
        print('\t\tEvaluating Train:')
        evaluate(batch_cache.train_batches, batch_cache.train_masks, 'train')
        print('\t\tEvaluating Validate:')
        m = evaluate(batch_cache.validate_batches, batch_cache.validate_masks, 'validate')
        print('\t\tEvaluating Test:')
        evaluate(batch_cache.test_batches, batch_cache.test_masks, 'test')
        scheduler.step(epoch)

        t1 = time.time()
        print('\tProcess Time: {}'.format(int(t1 - t0)))
        logs[-1].update({'process_time': t1 - t0})

        if m < best_metric:
            best_metric = m
            best_epoch = epoch
            print(f'\tSaving Model...')
            torch.save(model.state_dict(), f'{MODEL_DICT_DIR}/{tag}-model.pkl')
        logs[-1].update({'best_epoch': best_epoch})
        save_log(logs,
                 directory='QM7' if dataset == QMDataset.QM7
                 else 'QM8' if dataset == QMDataset.QM8
                 else 'QM9',
                 tag=tag)
