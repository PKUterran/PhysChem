import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from typing import List, Dict, Tuple
from itertools import chain
from functools import reduce
from tqdm import tqdm

from data.TOX21.load_tox21 import load_tox21
from net.config import ConfType
from net.models import GeomNN, RecGeomNN
from net.components import MLP
from .config import TOX21_CONFIG
from .utils.cache_batch import Batch, BatchCache, load_batch_cache, load_encode_mols, batch_cuda_copy
from .utils.seed import set_seed
from .utils.loss_functions import multi_roc
from .utils.save_log import save_log


def train_tox21(special_config: dict = None,
                use_cuda=False, max_num=-1, data_name='TOX21', seed=0, force_save=False, tag='TOX21',
                use_tqdm=False):
    # set parameters and seed
    print(f'For {tag}:')
    config = TOX21_CONFIG.copy()
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
    mols, mol_properties = load_tox21(max_num)
    mols_info = load_encode_mols(mols, name=data_name)

    # cache batches
    print('Caching Batches...')
    try:
        batch_cache = load_batch_cache(data_name, mols, mols_info, mol_properties, batch_size=config['BATCH'],
                                       needs_rdkit_conf=rdkit_support, contains_ground_truth_conf=False,
                                       use_cuda=use_cuda, use_tqdm=use_tqdm, force_save=force_save)
    except EOFError:
        batch_cache = load_batch_cache(data_name, mols, mols_info, mol_properties, batch_size=config['BATCH'],
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
        out_dim=12,
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
    loss_func = nn.BCEWithLogitsLoss()

    def nan_masked(s: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        nan_mask = torch.isnan(t)
        s[nan_mask] = -1e6
        t[nan_mask] = 0
        return s, t

    def nan_masked_np(s: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nan_mask = np.isnan(t)
        s[nan_mask] = -1e6
        t[nan_mask] = 0
        return s, t

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
            pred_p, properties = nan_masked(pred_p, batch.properties)
            p_loss = loss_func(pred_p, properties)
            loss = p_loss
            loss.backward()
            optimizer.step()

    def evaluate(batches: List[Batch], batch_name: str):
        model.eval()
        classifier.eval()
        optimizer.zero_grad()
        n_batch = len(batches)
        list_loss = []
        list_pred_p = []
        list_properties = []
        if use_tqdm:
            batches = tqdm(batches, total=n_batch)
        for batch in batches:
            if use_cuda:
                batch = batch_cuda_copy(batch)
            fp, _ = model.forward(batch.atom_ftr, batch.bond_ftr, batch.massive, batch.mask_matrices,
                                  batch.rdkit_conf)
            pred_p = classifier.forward(fp)
            pred_p, properties = nan_masked(pred_p, batch.properties)
            p_loss = loss_func(pred_p, properties)
            loss = p_loss
            list_loss.append(loss.cpu().item())

            list_pred_p.append(pred_p.cpu().detach().numpy())
            list_properties.append(batch.properties.cpu().numpy())
        pred_p = np.vstack(list_pred_p)
        properties = np.vstack(list_properties)
        pred_p, properties = nan_masked_np(pred_p, properties)
        p_total_roc = multi_roc(pred_p, properties)

        print(f'\t\t\tLOSS: {sum(list_loss) / n_batch}')
        print(f'\t\t\tMULTI-ROC: {p_total_roc}')
        logs[-1].update({
            f'{batch_name}_loss': sum(list_loss) / n_batch,
            f'{batch_name}_p_metric': p_total_roc,
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
        save_log(logs, directory='TOX21', tag=tag)