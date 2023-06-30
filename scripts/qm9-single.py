from net.config import ConfType
from train.train_qm9 import train_qm9


train_qm9(
    special_config={
        'CONF_TYPE': ConfType.SINGLE_CHANNEL,
    },
    use_cuda=True,
    max_num=-1,
    data_name='QM9-single',
    seed=0,
    force_save=False,
    tag='QM9-single',
    use_tqdm=False,
)
