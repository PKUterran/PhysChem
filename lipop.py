from train.train_lipop import train_lipop


train_lipop(
    special_config={},
    use_cuda=True,
    max_num=-1,
    data_name='Lipop',
    seed=0,
    force_save=False,
    tag='Lipop',
    use_tqdm=False
)
