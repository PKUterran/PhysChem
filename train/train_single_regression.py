import time
import torch
import torch.optim as optim
import numpy as np

from enum import Enum
from typing import List, Dict
from itertools import chain
from functools import reduce
from tqdm import tqdm

from data.Lipop.load_lipop import load_lipop
from data.ESOL.load_esol import load_esol
from data.FreeSolv.load_freesolv import load_freesolv
from net.config import ConfType
from net.models import GeomNN, RecGeomNN
from net.components import MLP
from .config import LIPOP_CONFIG, ESOL_CONFIG, FREESOLV_CONFIG
from .utils.cache_batch import Batch, BatchCache, load_batch_cache, load_encode_mols, batch_cuda_copy
from .utils.seed import set_seed
from .utils.loss_functions import mse_loss, rmse_loss
from .utils.save_log import save_log


class SingleRegressionDataset(Enum):
    LIPOP = 1,
    ESOL = 2,
    FREESOLV = 3,


def train_single_regression(
        dataset, data_name, tag,
        special_config: dict = None,
        use_cuda=False, max_num=-1, seed=0, force_save=False,
        use_tqdm=False):
    # set parameters and seed
    print(f'For {tag}:')
    if dataset == SingleRegressionDataset.LIPOP:
        config = LIPOP_CONFIG.copy()
    elif dataset == SingleRegressionDataset.ESOL:
        config = ESOL_CONFIG.copy()
    elif dataset == SingleRegressionDataset.FREESOLV:
        config = FREESOLV_CONFIG.copy()
    else:
        assert False
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
    if dataset == SingleRegressionDataset.LIPOP:
        mols, mol_properties = load_lipop(max_num)
    elif dataset == SingleRegressionDataset.ESOL:
        mols, mol_properties = load_esol(max_num)
    elif dataset == SingleRegressionDataset.FREESOLV:
        mols, mol_properties = load_freesolv(max_num)
    else:
        assert False
    mols_info = load_encode_mols(mols, name=data_name)

    # normalize properties and cache batches
    mean_p = np.mean(mol_properties, axis=0)
    stddev_p = np.std(mol_properties.tolist(), axis=0, ddof=1)
    norm_p = (mol_properties - mean_p) / stddev_p
    print('Caching Batches...')
    try:
        batch_cache = load_batch_cache(data_name, mols, mols_info, norm_p, batch_size=config['BATCH'],
                                       needs_rdkit_conf=rdkit_support, contains_ground_truth_conf=False,
                                       use_cuda=use_cuda, use_tqdm=use_tqdm, force_save=force_save)
    except EOFError:
        batch_cache = load_batch_cache(data_name, mols, mols_info, norm_p, batch_size=config['BATCH'],
                                       needs_rdkit_conf=rdkit_support, contains_ground_truth_conf=False,
                                       use_cuda=use_cuda, use_tqdm=use_tqdm, force_save=True)

    # build model
    print('Building Models...')
    atom_dim = batch_cache.atom_dim
    bond_dim = batch_cache.bond_dim
    model = RecGeomNN(
        atom_dim=atom_dim,
        bond_dim=bond_dim,
        config=config,
        use_cuda=use_cuda
    )
    classifier = MLP(
        in_dim=config['HM_DIM'],
        out_dim=1,
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

    def train(batches: List[Batch]):
        model.train()
        classifier.train()
        optimizer.zero_grad()
        n_batch = len(batches)
        if use_tqdm:
            batches = tqdm(batches, total=n_batch)
        for i, batch in enumerate(batches):
            if use_cuda:
                batch = batch_cuda_copy(batch)
            fp, _ = model.forward(batch.atom_ftr, batch.bond_ftr, batch.massive, batch.mask_matrices,
                                  batch.rdkit_conf)
            pred_p = classifier.forward(fp)
            p_loss = mse_loss(pred_p, batch.properties)
            loss = p_loss
            loss.backward()
            optimizer.step()

    def evaluate(batches: List[Batch], batch_name: str):
        model.eval()
        classifier.eval()
        optimizer.zero_grad()
        n_batch = len(batches)
        list_loss = []
        list_p_rmse = []
        if use_tqdm:
            batches = tqdm(batches, total=n_batch)
        for batch in batches:
            if use_cuda:
                batch = batch_cuda_copy(batch)
            fp, _ = model.forward(batch.atom_ftr, batch.bond_ftr, batch.massive, batch.mask_matrices,
                                  batch.rdkit_conf)
            pred_p = classifier.forward(fp)
            p_loss = mse_loss(pred_p, batch.properties)
            loss = p_loss
            list_loss.append(loss.cpu().item())

            p_rmse = rmse_loss(pred_p, batch.properties)
            list_p_rmse.append(p_rmse.item() * stddev_p[0])

        print(f'\t\t\tLOSS: {sum(list_loss) / n_batch}')
        print(f'\t\t\tRMSE: {sum(list_p_rmse) / n_batch}')
        logs[-1].update({
            f'{batch_name}_loss': sum(list_loss) / n_batch,
            f'{batch_name}_p_metric': sum(list_p_rmse) / n_batch,
        })

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
        evaluate(batch_cache.validate_batches, 'validate')
        print('\t\tEvaluating Test:')
        evaluate(batch_cache.test_batches, 'test')
        scheduler.step(epoch)

        t1 = time.time()
        print('\tProcess Time: {}'.format(int(t1 - t0)))
        logs[-1].update({'process_time': t1 - t0})
        save_log(logs,
                 directory='Lipop' if dataset == SingleRegressionDataset.LIPOP
                 else 'ESOL' if dataset == SingleRegressionDataset.ESOL
                 else 'FreeSolv',
                 tag=tag)