from train.train_qm9 import train_qm9

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='')
parser.add_argument('--lam', type=float, default=100.)
parser.add_argument('--iteration', type=int, default=4)
arg = parser.parse_args()
name = arg.name
lam = arg.lam
iteration = arg.iteration

train_qm9(
    special_config={
        'N_ITERATION': iteration,
        'LAMBDA': lam,
    },
    use_cuda=True,
    max_num=-1,
    data_name='QM9',
    seed=0,
    force_save=False,
    tag=f'ps-{name}',
    use_tqdm=False,
)
