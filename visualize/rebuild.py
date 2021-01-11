import torch
from typing import Tuple

from net.models import GeomNN, MLP
from train.config import QM9_CONFIG


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
