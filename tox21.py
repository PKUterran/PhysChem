from net.config import ConfType
from train.train_multi_classification import train_multi_classification, MultiClassificationDataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=16880611)
parser.add_argument('--pos', type=int, default=0)
arg = parser.parse_args()
seed = arg.seed
pos = arg.pos

extra_config = {}
if pos == 0:
    name = 'TOX21-Xconf'
    conf_type = ConfType.NONE
elif pos == 1:
    name = 'TOX21'
    conf_type = ConfType.RDKIT
else:
    name = 'TOX21-RGT'
    conf_type = ConfType.NEWTON_RGT
    extra_config = {}

train_multi_classification(
    dataset=MultiClassificationDataset.TOX21,
    data_name=f'{name}@{seed}',
    tag=f'{name}@{seed}',
    special_config=dict({
        'CONF_TYPE': conf_type,
    }, **extra_config),
    use_cuda=True,
    max_num=-1,
    seed=seed,
    force_save=False,
    use_tqdm=False
)
