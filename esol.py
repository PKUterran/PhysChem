from net.config import ConfType
from train.train_single_regression import train_single_regression, SingleRegressionDataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--pos', type=int, default=0)
arg = parser.parse_args()
seed = arg.seed
pos = arg.pos

train_single_regression(
    dataset=SingleRegressionDataset.ESOL,
    data_name='ESOL@{}'.format(seed) if pos else 'ESOL-Xconf@{}'.format(seed),
    tag='ESOL-M@{}'.format(seed) if pos else 'ESOL-Xconf@{}'.format(seed),
    special_config={
        'CONF_TYPE': ConfType.RDKIT if pos else ConfType.NONE,
    },
    use_cuda=True,
    max_num=-1,
    seed=seed,
    force_save=True,
    use_tqdm=False
)
