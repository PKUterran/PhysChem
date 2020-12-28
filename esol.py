from net.config import ConfType
from train.train_single_regression import train_single_regression, SingleRegressionDataset

train_single_regression(
    dataset=SingleRegressionDataset.ESOL,
    data_name='ESOL',
    tag='ESOL',
    special_config={

    },
    use_cuda=True,
    max_num=-1,
    seed=2,
    force_save=True,
    use_tqdm=False
)
train_single_regression(
    dataset=SingleRegressionDataset.ESOL,
    data_name='ESOL-Xconf',
    tag='ESOL-Xconf',
    special_config={
        'CONF_TYPE': ConfType.NONE,
    },
    use_cuda=True,
    max_num=-1,
    seed=2,
    force_save=True,
    use_tqdm=False
)
