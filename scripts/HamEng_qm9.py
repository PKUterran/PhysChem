from train.HamEng.train_qm9 import train_qm9, QMDataset

train_qm9(
    special_config={
        'EPOCH': 20,
    },
    dataset=QMDataset.QM9,
    use_cuda=True,
    max_num=-1,
    data_name='HamEng-QM9',
    seed=0,
    force_save=True,
    tag='HamEng-QM9',
    use_tqdm=False,
)
