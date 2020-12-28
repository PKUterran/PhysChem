from net.config import ConfType
from train.train_tox21 import train_tox21

train_tox21(
    special_config={

    },
    use_cuda=True,
    max_num=-1,
    data_name='TOX21',
    seed=0,
    force_save=True,
    tag='TOX21',
    use_tqdm=False
)
train_tox21(
    special_config={
        'CONF_TYPE': ConfType.NONE,
    },
    use_cuda=True,
    max_num=-1,
    data_name='TOX21-Xconf',
    seed=0,
    force_save=True,
    tag='TOX21-Xconf',
    use_tqdm=False
)
