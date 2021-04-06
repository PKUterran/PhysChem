import torch
from typing import Tuple

from net.models import GeomNN, MLP
from train.config import QM9_CONFIG
from train.CVGAE.config import QM9_CONFIG as CVGAE_CONFIG
from train.HamEng.config import FITTER_CONFIG_QM9 as HAMENG_CONFIG
from net.baseline.CVGAE.PredX_MPNN import CVGAE
from net.baseline.HamEng.models import HamiltonianPositionProducer


def rebuild_qm9(atom_dim, bond_dim, tag='QM9', special_config: dict = None, use_cuda=False) -> Tuple[GeomNN, MLP]:
    print(f'For {tag}:')
    config = QM9_CONFIG.copy()
    if special_config is not None:
        config.update(special_config)
    print('\t CONFIG:')
    for k, v in config.items():
        print(f'\t\t{k}: {v}')

    model = GeomNN(
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
    model_dicts = torch.load(f'train/models/{tag}-model.pkl', map_location=torch.device('cpu'))
    classifier_dicts = torch.load(f'train/models/{tag}-classifier.pkl', map_location=torch.device('cpu'))
    model.load_state_dict(model_dicts)
    classifier.load_state_dict(classifier_dicts)
    model.eval()
    classifier.eval()

    return model, classifier


def rebuild_cvgae(atom_dim, bond_dim, tag='CVGAE-QM9-rdkit', special_config: dict = None, use_cuda=False
                  ) -> [CVGAE, MLP]:
    print(f'For {tag}:')
    config = CVGAE_CONFIG.copy()
    if special_config is not None:
        config.update(special_config)
    print('\t CONFIG:')
    for k, v in config.items():
        print(f'\t\t{k}: {v}')

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
    model_dicts = torch.load(f'train/models/CVGAE/{tag}-model.pkl', map_location=torch.device('cpu'))
    conf_dicts = torch.load(f'train/models/CVGAE/{tag}-conf_gen.pkl', map_location=torch.device('cpu'))
    model.load_state_dict(model_dicts)
    conf_gen.load_state_dict(conf_dicts)
    model.eval()
    conf_gen.eval()

    return model, conf_gen


def rebuild_hameng(atom_dim, bond_dim, tag='HamEng-QM9', special_config: dict = None, use_cuda=False
                   ) -> Tuple[HamiltonianPositionProducer, MLP]:
    print(f'For {tag}:')
    config = HAMENG_CONFIG.copy()
    if special_config is not None:
        config.update(special_config)
    print('\t CONFIG:')
    for k, v in config.items():
        print(f'\t\t{k}: {v}')

    model = HamiltonianPositionProducer(
        n_dim=atom_dim,
        e_dim=bond_dim,
        config=config,
        use_cuda=use_cuda
    )
    conf_gen = MLP(
        in_dim=config['PQ_DIM'],
        out_dim=3,
        use_cuda=use_cuda,
        bias=False
    )
    model_dicts = torch.load(f'train/models/HamNet/{tag}-model.pkl', map_location=torch.device('cpu'))
    conf_dicts = torch.load(f'train/models/HamNet/{tag}-conf_gen.pkl', map_location=torch.device('cpu'))
    model.load_state_dict(model_dicts)
    conf_gen.load_state_dict(conf_dicts)
    model.eval()
    conf_gen.eval()

    return model, conf_gen
