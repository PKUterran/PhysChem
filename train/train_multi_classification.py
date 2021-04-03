import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from enum import Enum
from typing import List, Dict, Tuple
from itertools import chain
from functools import reduce
from tqdm import tqdm

from data.TOX21.load_tox21 import load_tox21
from data.sars.load_sars import load_sars
from net.config import ConfType
from net.models import GeomNN
from net.components import MLP
from .config import TOX21_CONFIG, SARS_CONFIG
from .utils.cache_batch import Batch, BatchCache, load_batch_cache, load_encode_mols, batch_cuda_copy
from .utils.seed import set_seed
from .utils.loss_functions import multi_roc, hierarchical_adj3_loss, distance_loss
from .utils.save_log import save_log


class MultiClassificationDataset(Enum):
    TOX21 = 1,
    SARS_COV_2 = 2,


def train_multi_classification(special_config: dict = None, dataset=MultiClassificationDataset.TOX21,
                               use_cuda=False, max_num=-1, data_name='TOX21', seed=0, force_save=False, tag='TOX21',
                               use_tqdm=False):
    # set parameters and seed
    print(f'For {tag}:')
    if dataset == MultiClassificationDataset.TOX21:
        config = TOX21_CONFIG.copy()
    else:
        config = SARS_CONFIG.copy()
    if special_config is not None:
        config.update(special_config)
    print('\t CONFIG:')
    for k, v in config.items():
        print(f'\t\t{k}: {v}')
    rdkit_support = config['CONF_TYPE'] == ConfType.RDKIT or config['CONF_TYPE'] == ConfType.NEWTON_RGT
    conf_supervised = config['CONF_TYPE'] == ConfType.NEWTON_RGT
    set_seed(seed, use_cuda=use_cuda)
    np.set_printoptions(suppress=True, precision=3, linewidth=200)

    # load dataset
    print('Loading:')
    if dataset == MultiClassificationDataset.TOX21:
        mols, mol_properties = load_tox21(max_num, force_save=force_save)
        n_class = 2
    else:
        mols, mol_properties = load_sars(max_num, force_save=force_save)
        n_class = 4
    mols_info = load_encode_mols(mols, name=data_name, force_save=force_save)

    # label normalization
    n_label = mol_properties.shape[1]
    cnt_notnan = []
    cnt_label_class = np.ones(shape=[n_label, n_class], dtype=np.float)
    for i in range(n_label):
        labels_i = mol_properties[:, i]
        labels_i = labels_i[np.logical_not(np.isnan(labels_i))]
        cnt_notnan.append(len(labels_i))
        n_class_b = len(set(labels_i))
        # assert n_class == n_class_b, f'{n_class} vs {n_class_b}'
        for label in labels_i:
            cnt_label_class[i][int(label)] += 1.
    weight_label_class = (np.repeat(np.expand_dims(cnt_notnan, -1), n_class, axis=-1) / n_class) * cnt_label_class ** -1
    print(f'\t\tLabel-Class: \n{cnt_label_class}')
    print(f'\t\tWeights: \n{weight_label_class}')
    weight_label_class = torch.tensor(weight_label_class, dtype=torch.float32)
    if use_cuda:
        weight_label_class = weight_label_class.cuda()

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
    model = GeomNN(
        atom_dim=atom_dim,
        bond_dim=bond_dim,
        config=config,
        use_cuda=use_cuda
    )
    classifiers = [MLP(
        in_dim=config['HM_DIM'],
        out_dim=n_class,
        activation='softmax',
        hidden_dims=config['CLASSIFIER_HIDDENS'],
        use_cuda=use_cuda,
        bias=True
    ) for _ in range(n_label)]
    if use_cuda:
        model.cuda()
        for c in classifiers:
            c.cuda()

    # initialize optimization
    parameters = list(chain(model.parameters(), *[c.parameters() for c in classifiers]))
    optimizer = optim.Adam(params=parameters, lr=config['LR'], weight_decay=config['DECAY'])
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=config['GAMMA'])
    print('##### Parameters #####')

    param_size = 0
    for name, param in chain(model.named_parameters(), *[c.named_parameters() for c in classifiers]):
        print(f'\t\t{name}: {param.shape}')
        param_size += reduce(lambda x, y: x * y, param.shape)
    print(f'\tNumber of parameters: {param_size}')

    # train
    epoch = 0
    logs: List[Dict[str, float]] = []
    loss_funcs = [nn.CrossEntropyLoss(weight=weight_label_class[i]) for i in range(n_label)]
    c_loss_fuc = hierarchical_adj3_loss

    def nan_masked(s: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        notnan_mask = ~torch.isnan(t)
        s = s[notnan_mask, :]
        t = t[notnan_mask]
        return s, t

    def train(batches: List[Batch]):
        model.train()
        for cls in classifiers:
            cls.train()
        optimizer.zero_grad()
        losses = []
        n_batch = len(batches)
        if use_tqdm:
            batches = tqdm(batches, total=n_batch)
        for i, batch in enumerate(batches):
            if use_cuda:
                batch = batch_cuda_copy(batch)
            fp, pred_cs, *_ = model.forward(batch.atom_ftr, batch.bond_ftr, batch.massive, batch.mask_matrices,
                                            batch.rdkit_conf)
            p_losses = []
            for j, cls in enumerate(classifiers):
                pred_p = cls.forward(fp)
                pred_p_, properties_ = nan_masked(pred_p, batch.properties[:, j])
                if pred_p_.shape[0] == 0:
                    continue
                p_losses.append(loss_funcs[j](pred_p_, properties_.type(torch.long)))
            p_loss = sum(p_losses)
            if conf_supervised:
                c_loss = c_loss_fuc(pred_cs, batch.conformation, batch.mask_matrices, use_cuda=use_cuda)
                loss = p_loss + config['LAMBDA'] * c_loss
            else:
                loss = p_loss
            # loss.backward()
            # optimizer.step()
            losses.append(loss)
            if len(losses) >= config['PACK'] or i == n_batch - 1:
                (sum(losses) / len(losses)).backward()
                optimizer.step()
                losses.clear()

    def evaluate(batches: List[Batch], batch_name: str):
        model.eval()
        for cls in classifiers:
            cls.train()
        optimizer.zero_grad()
        n_batch = len(batches)
        list_loss = []
        list_preds_p = [[] for _ in range(n_label)]
        list_properties = []
        list_c_loss = []
        list_rsd = []
        if use_tqdm:
            batches = tqdm(batches, total=n_batch)
        for batch in batches:
            if use_cuda:
                batch = batch_cuda_copy(batch)
            fp, pred_cs, *_ = model.forward(batch.atom_ftr, batch.bond_ftr, batch.massive, batch.mask_matrices,
                                            batch.rdkit_conf)
            p_losses = []
            for j, cls in enumerate(classifiers):
                pred_p = cls.forward(fp)
                pred_p_, properties_ = nan_masked(pred_p, batch.properties[:, j])
                if pred_p_.shape[0] == 0:
                    continue
                p_losses.append(loss_funcs[j](pred_p_, properties_.type(torch.long)))
                list_preds_p[j].append(pred_p_.cpu().detach().numpy())
            p_loss = sum(p_losses)
            if conf_supervised:
                c_loss = c_loss_fuc(pred_cs, batch.conformation, batch.mask_matrices, use_cuda=use_cuda)
                loss = p_loss + config['LAMBDA'] * c_loss
                rsd = distance_loss(pred_cs[-1], batch.conformation, batch.mask_matrices, root_square=True)
                list_rsd.append(rsd.cpu().item())
                list_c_loss.append(loss.cpu().item())
            else:
                loss = p_loss
            list_loss.append(loss.cpu().item())

            list_properties.append(batch.properties.cpu().numpy())
        list_pred_p = [np.vstack(preds_p) for preds_p in list_preds_p]
        properties = np.vstack(list_properties)
        p_total_roc, p_multi_roc = multi_roc(list_pred_p, properties)

        print(f'\t\t\tLOSS: {sum(list_loss) / n_batch}')
        print(f'\t\t\tAVG-ROC: {p_total_roc}')
        print(f'\t\t\tMULTI-ROC: {p_multi_roc}')
        logs[-1].update({
            f'{batch_name}_loss': sum(list_loss) / n_batch,
            f'{batch_name}_p_metric': p_total_roc,
            f'{batch_name}_p_multi_metric': p_multi_roc,
        })
        if conf_supervised:
            print(f'\t\t\tC LOSS: {sum(list_c_loss) / n_batch}')
            print(f'\t\t\tDL-RS: {sum(list_rsd) / n_batch}')
            logs[-1].update({
                f'{batch_name}_c_loss': sum(list_c_loss) / n_batch,
                f'{batch_name}_c_metric': sum(list_rsd) / n_batch,
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
        save_log(logs, directory='TOX21' if dataset == MultiClassificationDataset.TOX21 else 'sars', tag=tag)
