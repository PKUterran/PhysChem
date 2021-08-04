import os
import time
import torch
import torch.optim as optim
import numpy as np

from enum import Enum
from typing import List, Dict
from itertools import chain
from functools import reduce
from tqdm import tqdm

# from data.geom_qm9.load_qm9 import load_qm9 as load_geom_qm9
from data.qm7.load_qm7 import load_qm7
from data.qm8.load_qm8 import load_qm8
from data.qm9.load_qm9 import load_qm9
from net.config import ConfType
from net.models import GeomNN, MLP
from .config import QM9_CONFIG
from .utils.cache_batch import Batch, load_batch_cache, load_encode_mols, batch_cuda_copy
from .utils.seed import set_seed
from .utils.loss_functions import multi_mse_loss, multi_mae_loss, adj3_loss, distance_loss, \
    hierarchical_adj2_loss, hierarchical_adj3_loss, hierarchical_adj4_loss, kabsch_rmsd_loss, \
    hierarchical_mixed_kabsch_adj3_loss
from .utils.save_log import save_log

MODEL_DICT_DIR = 'train/models'


class QMDataset(Enum):
    QM7 = 1,
    QM8 = 2,
    QM9 = 3,


def train_qm9(special_config: dict = None, dataset=QMDataset.QM9,
              use_cuda=False, max_num=-1, data_name='QM9', seed=0, force_save=False, tag='QM9',
              use_tqdm=False):
    # set parameters and seed
    print(f'For {tag}:')
    config = QM9_CONFIG.copy()
    if special_config is not None:
        config.update(special_config)
    print('\t CONFIG:')
    for k, v in config.items():
        print(f'\t\t{k}: {v}')
    real_support = config['CONF_TYPE'] == ConfType.REAL
    rdkit_support = config['CONF_TYPE'] == ConfType.RDKIT or config['CONF_TYPE'] == ConfType.NEWTON_RGT
    rdkit_groundtruth = config['CONF_TYPE'] == ConfType.NEWTON_RGT
    conf_only = config['CONF_TYPE'] == ConfType.ONLY
    set_seed(seed, use_cuda=use_cuda)
    np.set_printoptions(suppress=True, precision=3, linewidth=200)

    # load dataset
    print('Loading:')
    # mol_list_weight_mol, mol_properties = load_geom_qm9(max_num)
    # mols = [list_weight_mol[0][1] for list_weight_mol in mol_list_weight_mol]
    if dataset == QMDataset.QM7:
        mols, mol_properties = load_qm7(max_num)
        mols_info = load_encode_mols(mols, name=data_name, force_save=force_save)
    elif dataset == QMDataset.QM8:
        mols, mol_properties = load_qm8(max_num)
        mols_info = load_encode_mols(mols, name=data_name, force_save=force_save)
    else:
        mols, mol_properties = load_qm9(max_num)
        mols_info = load_encode_mols(mols, name=data_name, force_save=force_save)

    # normalize properties and cache batches
    mean_p = np.mean(mol_properties, axis=0)
    stddev_p = np.std((mol_properties - mean_p).tolist(), axis=0, ddof=0)
    weights = torch.tensor(stddev_p, dtype=torch.float32) * 22
    if use_cuda:
        weights = weights.cuda()
    # mad_p = np.array([1.189, 6.299, 0.016, 0.039, 0.040, 202.017,
    #                   0.026, 31.072, 31.072, 31.072, 31.072, 3.204], dtype=np.float)
    norm_p = (mol_properties - mean_p) / stddev_p
    print(f'\tmean: {mean_p}')
    print(f'\tstd: {stddev_p}')
    # print(f'\tmad: {mad_p}')
    if dataset == QMDataset.QM8:
        print(f'\tweights: {weights.cpu().numpy()}')
    print('Caching Batches...')
    try:
        batch_cache = load_batch_cache(data_name, mols, mols_info, norm_p, batch_size=config['BATCH'],
                                       needs_rdkit_conf=rdkit_support, contains_ground_truth_conf=not rdkit_groundtruth,
                                       use_cuda=use_cuda, use_tqdm=use_tqdm, force_save=force_save)
    except EOFError:
        batch_cache = load_batch_cache(data_name, mols, mols_info, norm_p, batch_size=config['BATCH'],
                                       needs_rdkit_conf=rdkit_support, contains_ground_truth_conf=not rdkit_groundtruth,
                                       use_cuda=use_cuda, use_tqdm=use_tqdm, force_save=True)

    # build model
    print('Building Models...')
    atom_dim = batch_cache.atom_dim
    bond_dim = batch_cache.bond_dim
    model = GeomNN(
        atom_dim=atom_dim,
        bond_dim=bond_dim,
        config=config,
        use_cuda=use_cuda
    )
    classifier = MLP(
        in_dim=config['HM_DIM'],
        out_dim=mol_properties.shape[1],
        hidden_dims=config['CLASSIFIER_HIDDENS'],
        use_cuda=use_cuda,
        bias=True
    )
    if use_cuda:
        model.cuda()
        classifier.cuda()

    # initialize optimization
    parameters = list(chain(model.parameters(), classifier.parameters()))
    optimizer = optim.Adam(params=parameters, lr=config['LR'], weight_decay=config['DECAY'])
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=config['GAMMA'])
    print('##### Parameters #####')

    param_size = 0
    for name, param in chain(model.named_parameters(), classifier.named_parameters()):
        print(f'\t\t{name}: {param.shape}')
        param_size += reduce(lambda x, y: x * y, param.shape)
    print(f'\tNumber of parameters: {param_size}')

    # train
    epoch = 0
    logs: List[Dict[str, float]] = []
    best_epoch = 0
    best_metric = 999
    if config['CONF_LOSS'] == 'DL':
        c_loss_fuc = distance_loss
    elif config['CONF_LOSS'] == 'ADJ3':
        c_loss_fuc = adj3_loss
    elif config['CONF_LOSS'] == 'H_ADJ2':
        c_loss_fuc = hierarchical_adj2_loss
    elif config['CONF_LOSS'] == 'H_ADJ3':
        c_loss_fuc = hierarchical_adj3_loss
    elif config['CONF_LOSS'] == 'H_ADJ4':
        c_loss_fuc = hierarchical_adj4_loss
    elif config['CONF_LOSS'] == 'Kabsch':
        c_loss_fuc = kabsch_rmsd_loss
    elif config['CONF_LOSS'] == 'H_mixed':
        c_loss_fuc = hierarchical_mixed_kabsch_adj3_loss
    else:
        assert False, f"Unknown conformation loss function: {config['CONF_LOSS']}"
    try:
        if not os.path.exists(MODEL_DICT_DIR):
            os.mkdir(MODEL_DICT_DIR)
    except FileExistsError:
        pass

    def train(batches: List[Batch]):
        model.train()
        classifier.train()
        optimizer.zero_grad()
        n_batch = len(batches)
        if use_tqdm:
            batches = tqdm(batches, total=n_batch)
        for batch in batches:
            if use_cuda:
                batch = batch_cuda_copy(batch)
            fp, pred_cs, *_ = model.forward(batch.atom_ftr, batch.bond_ftr, batch.massive, batch.mask_matrices,
                                            batch.rdkit_conf if not real_support else batch.conformation)
            pred_p = classifier.forward(fp)
            if dataset == QMDataset.QM7:
                p_loss = multi_mae_loss(pred_p, batch.properties)
            elif dataset == QMDataset.QM8:
                p_losses = multi_mae_loss(pred_p, batch.properties, explicit=True)
                p_loss = sum(p_losses * weights)
            else:
                p_loss = multi_mse_loss(pred_p, batch.properties)
            if config['CONF_LOSS'].startswith('H_'):
                c_loss = c_loss_fuc(pred_cs, batch.conformation, batch.mask_matrices, use_cuda=use_cuda)
            else:
                c_loss = c_loss_fuc(pred_cs[-1], batch.conformation, batch.mask_matrices, use_cuda=use_cuda)
            if conf_only:
                loss = config['LAMBDA'] * c_loss
            else:
                loss = p_loss + config['LAMBDA'] * c_loss
            loss.backward()
            optimizer.step()

    def evaluate(batches: List[Batch], batch_name: str) -> float:
        model.eval()
        classifier.eval()
        optimizer.zero_grad()
        n_batch = len(batches)
        list_p_loss = []
        list_c_loss = []
        list_loss = []
        list_p_multi_mae = []
        list_p_total_mae = []
        list_rsd = []
        list_kabsch = []
        if use_tqdm:
            batches = tqdm(batches, total=n_batch)
        for batch in batches:
            if use_cuda:
                batch = batch_cuda_copy(batch)
            fp, pred_cs, *_ = model.forward(batch.atom_ftr, batch.bond_ftr, batch.massive, batch.mask_matrices,
                                            batch.rdkit_conf if not real_support else batch.conformation)
            pred_p = classifier.forward(fp)
            if dataset == QMDataset.QM7:
                p_loss = multi_mae_loss(pred_p, batch.properties)
            elif dataset == QMDataset.QM8:
                p_losses = multi_mae_loss(pred_p, batch.properties, explicit=True)
                p_loss = sum(p_losses * weights)
            else:
                p_loss = multi_mse_loss(pred_p, batch.properties)
            if config['CONF_LOSS'].startswith('H_'):
                c_loss = c_loss_fuc(pred_cs, batch.conformation, batch.mask_matrices, use_cuda=use_cuda)
            else:
                c_loss = c_loss_fuc(pred_cs[-1], batch.conformation, batch.mask_matrices, use_cuda=use_cuda)
            if conf_only:
                loss = config['LAMBDA'] * c_loss
            else:
                loss = p_loss + config['LAMBDA'] * c_loss
            list_p_loss.append(p_loss.cpu().item())
            list_c_loss.append(c_loss.cpu().item())
            list_loss.append(loss.cpu().item())

            p_multi_mae = multi_mae_loss(pred_p, batch.properties, explicit=True)
            p_total_mae = p_multi_mae.sum()
            rsd = distance_loss(pred_cs[-1], batch.conformation, batch.mask_matrices, root_square=True)
            list_p_multi_mae.append(p_multi_mae.cpu().detach().numpy())
            list_p_total_mae.append(p_total_mae.cpu().item())
            list_rsd.append(rsd.cpu().item())
            if config['CONF_LOSS'] in ['H_mixed', 'Kabsch']:
                kabsch = kabsch_rmsd_loss(pred_cs[-1], batch.conformation, batch.mask_matrices, use_cuda=use_cuda)
                list_kabsch.append(kabsch.cpu().item())

        print(f'\t\t\tP LOSS: {sum(list_p_loss) / n_batch}')
        print(f'\t\t\tC LOSS: {sum(list_c_loss) / n_batch}')
        print(f'\t\t\tTOTAL LOSS: {sum(list_loss) / n_batch}')
        print(f'\t\t\tPROPERTIES MULTI-MAE: {sum(list_p_multi_mae) * stddev_p / n_batch}')
        if dataset == QMDataset.QM8:
            total_mae = np.sum(sum(list_p_multi_mae) * stddev_p / n_batch)
        else:
            total_mae = sum(list_p_total_mae) / n_batch
        print(f'\t\t\tPROPERTIES TOTAL MAE: {total_mae}')
        print(f'\t\t\tCONFORMATION RS-DL: {sum(list_rsd) / n_batch}')
        if config['CONF_LOSS'] in ['H_mixed', 'Kabsch']:
            print(f'\t\t\tKABSCH RMSD: {sum(list_kabsch) / n_batch}')
            logs[-1].update({
                f'{batch_name}_kabsch': sum(list_kabsch) / n_batch,
            })
        logs[-1].update({
            f'{batch_name}_p_loss': sum(list_p_loss) / n_batch,
            f'{batch_name}_c_loss': sum(list_c_loss) / n_batch,
            f'{batch_name}_loss': sum(list_loss) / n_batch,
            f'{batch_name}_p_metric': total_mae,
            f'{batch_name}_multi_p_metric': list(sum(list_p_multi_mae) * stddev_p / n_batch),
            f'{batch_name}_c_metric': sum(list_rsd) / n_batch,
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
        train(batch_cache.train_batches)
        print('\t\tEvaluating Train:')
        evaluate(batch_cache.train_batches, 'train')
        print('\t\tEvaluating Validate:')
        m = evaluate(batch_cache.validate_batches, 'validate')
        print('\t\tEvaluating Test:')
        evaluate(batch_cache.test_batches, 'test')
        scheduler.step()

        t1 = time.time()
        print('\tProcess Time: {}'.format(int(t1 - t0)))
        logs[-1].update({'process_time': t1 - t0})

        if m < best_metric:
            best_metric = m
            best_epoch = epoch
            print(f'\tSaving Model...')
            torch.save(model.state_dict(), f'{MODEL_DICT_DIR}/{tag}-model.pkl')
            torch.save(classifier.state_dict(), f'{MODEL_DICT_DIR}/{tag}-classifier.pkl')
        logs[-1].update({'best_epoch': best_epoch})
        save_log(logs,
                 directory='QM7' if dataset == QMDataset.QM7
                 else 'QM8' if dataset == QMDataset.QM8
                 else 'QM9',
                 tag=tag)
