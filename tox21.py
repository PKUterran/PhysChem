from net.config import ConfType
from train.train_tox21 import train_tox21

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--pos', type=int, default=0)
arg = parser.parse_args()
seed = arg.seed
pos = arg.pos

if seed == 0:
    train_tox21(
        data_name='TOX21' if pos else 'TOX21-Xconf',
        tag='TOX21' if pos else 'TOX21-Xconf',
        special_config={
            'CONF_TYPE': ConfType.RDKIT if pos else ConfType.NONE,
        },
        use_cuda=True,
        max_num=-1,
        seed=seed,
        force_save=False,
        use_tqdm=False
    )
    exit(0)

train_tox21(
    data_name='TOX21@{}'.format(seed) if pos else 'TOX21-Xconf@{}'.format(seed),
    tag='TOX21@{}'.format(seed) if pos else 'TOX21-Xconf@{}'.format(seed),
    special_config={
        'CONF_TYPE': ConfType.RDKIT if pos else ConfType.NONE,
    },
    use_cuda=True,
    max_num=-1,
    seed=seed,
    force_save=True,
    use_tqdm=False
)
