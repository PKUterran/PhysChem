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
from net.models import MLP
from net.baseline.CVGAE.PredX_MPNN import CVGAE
from .config import QM9_CONFIG
from train.utils.cache_batch import Batch, load_batch_cache, load_encode_mols, batch_cuda_copy
from train.utils.seed import set_seed
from train.utils.loss_functions import multi_mse_loss, multi_mae_loss, adj3_loss, distance_loss
from train.utils.save_log import save_log

MODEL_DICT_DIR = 'train/models/CVGAE'


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
    rdkit_support = config['CONF_TYPE'] == ConfType.RDKIT
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
    # mad_p = np.array([1.189, 6.299, 0.016, 0.039, 0.040, 202.017,
    #                   0.026, 31.072, 31.072, 31.072, 31.072, 3.204], dtype=np.float32)
    norm_p = (mol_properties - mean_p) / stddev_p
    print(f'\tmean: {mean_p}')
    print(f'\tstd: {stddev_p}')
    # print(f'\tmad: {mad_p}')
    print('Caching Batches...')
    try:
        batch_cache = load_batch_cache(data_name, mols, mols_info, norm_p, batch_size=config['BATCH'],
                                       needs_rdkit_conf=rdkit_support,
                                       use_cuda=use_cuda, use_tqdm=use_tqdm, force_save=force_save)
    except EOFError:
        batch_cache = load_batch_cache(data_name, mols, mols_info, norm_p, batch_size=config['BATCH'],
                                       needs_rdkit_conf=rdkit_support,
                                       use_cuda=use_cuda, use_tqdm=use_tqdm, force_save=True)

    # build model
    print('Building Models...')
    atom_dim = batch_cache.atom_dim
    bond_dim = batch_cache.bond_dim
    model = CVGAE(
        atom_dim=atom_dim,
        bond_dim=bond_dim,
        config=config,
        use_cuda=use_cuda
    )
    conf_gen = MLP(
        in_dim=2 * config['HV_DIM'],
        out_dim=3,
        use_cuda=use_cuda,
        bias=False
    )
    if use_cuda:
        model.cuda()
        conf_gen.cuda()

    # initialize optimization
    parameters = list(chain(model.parameters(), conf_gen.parameters()))
    optimizer = optim.Adam(params=parameters, lr=config['LR'], weight_decay=config['DECAY'])
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=config['GAMMA'])
    print('##### Parameters #####')

    param_size = 0
    for name, param in chain(model.named_parameters(), conf_gen.named_parameters()):
        print(f'\t\t{name}: {param.shape}')
        param_size += reduce(lambda x, y: x * y, param.shape)
    print(f'\tNumber of parameters: {param_size}')

    # train
    epoch = 0
    logs: List[Dict[str, float]] = []
    best_epoch = 0
    best_metric = 999
    c_loss_fuc = adj3_loss
    try:
        if not os.path.exists(MODEL_DICT_DIR):
            os.mkdir(MODEL_DICT_DIR)
    except FileExistsError:
        pass

    def train(batches: List[Batch], batch_name: str = 'train'):
        model.train()
        conf_gen.train()
        optimizer.zero_grad()
        n_batch = len(batches)
        list_c_loss = []
        list_kz_loss = []
        list_k0_loss = []
        list_loss = []
        if use_tqdm:
            batches = tqdm(batches, total=n_batch)
        for batch in batches:
            if use_cuda:
                batch = batch_cuda_copy(batch)
            c_losses = []
            kz_losses = []
            k0_losses = []
            for _ in range(config['SAMPLE']):
                pred_c, klds_z, klds_0 = model.forward(batch.atom_ftr, batch.bond_ftr, batch.mask_matrices,
                                                       is_training=True, given_pos=batch.conformation)
                conf = conf_gen.forward(pred_c)
                c_loss = c_loss_fuc(conf, batch.conformation, batch.mask_matrices, use_cuda=use_cuda)
                kz_loss = torch.mean(klds_z)
                k0_loss = torch.mean(klds_0)
                c_losses.append(c_loss)
                kz_losses.append(kz_loss)
                k0_losses.append(k0_loss)
            t_c_loss = sum(c_losses) / config['SAMPLE']
            t_kz_loss = sum(kz_losses) / config['SAMPLE']
            t_k0_loss = sum(k0_losses) / config['SAMPLE']
            t_loss = t_c_loss + config['LAMBDA'] * t_kz_loss + config['LAMBDA'] * 1e-3 * t_k0_loss
            t_loss.backward()
            optimizer.step()

            list_c_loss.append(t_c_loss.cpu().item())
            list_kz_loss.append(t_kz_loss.cpu().item())
            list_k0_loss.append(t_k0_loss.cpu().item())
            list_loss.append(t_loss.cpu().item())
        print(f'\t\t\tC LOSS: {sum(list_c_loss) / n_batch}')
        print(f'\t\t\tKLD_Z LOSS: {sum(list_kz_loss) / n_batch}')
        print(f'\t\t\tKLD_0 LOSS: {sum(list_k0_loss) / n_batch}')
        print(f'\t\t\tTOTAL LOSS: {sum(list_loss) / n_batch}')
        logs[-1].update({
            f'{batch_name}_c_loss': sum(list_c_loss) / n_batch,
            f'{batch_name}_kz_loss': sum(list_kz_loss) / n_batch,
            f'{batch_name}_k0_loss': sum(list_k0_loss) / n_batch,
            f'{batch_name}_loss': sum(list_loss) / n_batch,
        })

    def evaluate(batches: List[Batch], batch_name: str) -> float:
        model.eval()
        conf_gen.eval()
        optimizer.zero_grad()
        n_batch = len(batches)
        list_rsd = []
        if use_tqdm:
            batches = tqdm(batches, total=n_batch)
        for batch in batches:
            if use_cuda:
                batch = batch_cuda_copy(batch)
            rsds = []
            for _ in range(config['SAMPLE']):
                pred_c = model.forward(batch.atom_ftr, batch.bond_ftr, batch.mask_matrices,
                                       is_training=False, given_pos=batch.rdkit_conf)
                conf = conf_gen.forward(pred_c)
                rsd = distance_loss(conf, batch.conformation, batch.mask_matrices, root_square=True)
                rsds.append(rsd)

            list_rsd.append((sum(rsds) / len(rsds)).cpu().item())

        print(f'\t\t\tCONFORMATION RS-DL: {sum(list_rsd) / n_batch}')
        logs[-1].update({
            f'{batch_name}_c_metric': sum(list_rsd) / n_batch,
        })
        return sum(list_rsd) / n_batch

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
        scheduler.step(epoch)

        t1 = time.time()
        print('\tProcess Time: {}'.format(int(t1 - t0)))
        logs[-1].update({'process_time': t1 - t0})

        if m < best_metric:
            best_metric = m
            best_epoch = epoch
            print(f'\tSaving Model...')
            torch.save(model.state_dict(), f'{MODEL_DICT_DIR}/{tag}-model.pkl')
            torch.save(conf_gen.state_dict(), f'{MODEL_DICT_DIR}/{tag}-conf_gen.pkl')
        logs[-1].update({'best_epoch': best_epoch})
        save_log(logs,
                 directory='QM7' if dataset == QMDataset.QM7
                 else 'QM8' if dataset == QMDataset.QM8
                 else 'QM9',
                 tag=tag)
