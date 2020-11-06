import torch
import torch.optim as optim
import numpy as np

from typing import List
from itertools import chain

from data.qm9.load_qm9 import load_qm9
from data.encode import encode_mols
from net.models import GeomNN
from net.components import MLP
from .config import QM9_CONFIG
from .utils.cache_batch import Batch, BatchCache
from .utils.seed import set_seed
from .utils.loss_functions import multi_mse_loss, multi_mae_loss, adj3_loss


def train_qm9(special_config: dict = None,
              use_cuda=False, max_num=-1, seed=0):
    # set parameters and seed
    config = QM9_CONFIG.copy()
    if special_config is not None:
        config.update(special_config)
    set_seed(seed, use_cuda=use_cuda)

    # load dataset
    mol_list_weight_mol, mol_properties = load_qm9(max_num)
    mols = [list_weight_mol[0][1] for list_weight_mol in mol_list_weight_mol]
    mols_info = encode_mols(mols)

    # normalize properties and cache batches
    mean_p = np.mean(mol_properties, axis=0)
    stddev_p = np.std(mol_properties, axis=0, ddof=1)
    norm_p = (mol_properties - mean_p) / stddev_p
    batch_cache = BatchCache(mols, mols_info, norm_p, use_cuda=use_cuda, batch_size=config)

    # build model
    atom_dim = batch_cache.train_batches[0].atom_ftr.shape[1]
    bond_dim = batch_cache.train_batches[0].bond_ftr.shape[1]
    model = GeomNN(
        atom_dim=atom_dim,
        bond_dim=bond_dim,
        config=config,
        use_cuda=use_cuda
    )
    classifier = MLP(
        in_dim=config['HM_DIM'],
        out_dim=12,
        hidden_dims=[128],
        use_cuda=use_cuda,
        bias=True
    )

    # initialize optimization
    parameters = list(chain(model.parameters(), classifier.parameters()))
    optimizer = optim.Adam(params=parameters, lr=config)

    # train

    def train(batches: List[Batch]):
        model.train()
        model.zero_grad()
        classifier.train()
        classifier.zero_grad()
        for batch in batches:
            fp, conf = model(batch.atom_ftr, batch.bond_ftr, batch.massive, batch.mask_matrices)
            pred_p = classifier(fp)
            p_loss = multi_mse_loss(pred_p, batch.properties)
            c_loss = adj3_loss(conf, batch.conformation, batch.mask_matrices)
            loss = p_loss + config['LAMBDA'] * c_loss
            loss.backward()
            optimizer.step()

    def evaluate():
        pass
