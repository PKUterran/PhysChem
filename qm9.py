from train.train_qm9 import train_qm9

train_qm9(
    special_config={
        # 'N_LAYER': 2,
        # 'N_HOP': 1,
        # 'N_ITERATION': 4,
        # 'N_GLOBAL': 2,
        # 'LAMBDA': 1,
        'CONF_LOSS': 'H_ADJ2',
    },
    use_cuda=True,
    max_num=-1,
    data_name='QM9',
    seed=0,
    force_save=False,
    tag='QM9-H_ADJ2',
    use_tqdm=False,
)
