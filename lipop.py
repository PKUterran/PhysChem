from net.config import ConfType
from train.train_single_regression import train_single_regression, SingleRegressionDataset

# train_lipop(
#     special_config={},
#     use_cuda=True,
#     max_num=-1,
#     data_name='Lipop',
#     seed=0,
#     force_save=False,
#     tag='Lipop',
#     use_tqdm=False
# )
train_single_regression(
    dataset=SingleRegressionDataset.LIPOP,
    data_name='Lipop-Xconf',
    tag='Lipop-Xconf',
    special_config={
        'CONF_TYPE': ConfType.NONE,
    },
    use_cuda=True,
    max_num=-1,
    seed=0,
    force_save=False,
    use_tqdm=False
)
