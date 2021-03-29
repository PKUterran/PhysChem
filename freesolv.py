from net.config import ConfType
from train.train_single_regression import train_single_regression, SingleRegressionDataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--pos', type=int, default=0)
arg = parser.parse_args()
seed = arg.seed
pos = arg.pos

extra_config = {}
if pos == 0:
    name = 'FreeSolv-Xconf'
    conf_type = ConfType.NONE
elif pos == 1:
    name = 'FreeSolv'
    conf_type = ConfType.RDKIT
else:
    name = 'FreeSolv-RGT'
    conf_type = ConfType.NEWTON_RGT
    extra_config = {'LR': 3e-4, 'BATCH': 2, 'PACK': 64}

train_single_regression(
    dataset=SingleRegressionDataset.FREESOLV,
    data_name=f'{name}@{seed}',
    tag=f'{name}@{seed}',
    special_config=dict({
        'CONF_TYPE': conf_type,
    }, **extra_config),
    use_cuda=True,
    max_num=-1,
    seed=seed,
    force_save=True,
    use_tqdm=False
)
