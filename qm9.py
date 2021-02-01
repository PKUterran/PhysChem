from train.train_qm9 import train_qm9


train_qm9(
    special_config={
        'LAMBDA': 10
    },
    use_cuda=True,
    max_num=-1,
    data_name='QM9',
    seed=0,
    force_save=False,
    tag='QM9-lambda10',
    use_tqdm=False,
)
