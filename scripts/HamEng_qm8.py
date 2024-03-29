from train.HamEng.train_qm9 import train_qm9, QMDataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=16880611)
arg = parser.parse_args()
seed = arg.seed

train_qm9(
    special_config={
        'LR': 3e-5,
    },
    dataset=QMDataset.QM8,
    use_cuda=True,
    max_num=-1,
    data_name=f'HamEng@{seed}',
    seed=seed,
    force_save=True,
    tag=f'HamEng@{seed}',
    use_tqdm=False,
)
