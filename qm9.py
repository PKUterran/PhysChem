from train.train_qm9 import train_qm9


train_qm9(
    special_config={

    },
    use_cuda=True,
    max_num=-1,
    data_name='QM9',
    seed=0,
    force_save=True,
    tag='QM9',
    use_tqdm=False,
)
