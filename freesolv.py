from net.config import ConfType
from train.train_single_regression import train_single_regression, SingleRegressionDataset

train_single_regression(
    dataset=SingleRegressionDataset.FREESOLV,
    data_name='FreeSolv',
    tag='FreeSolv',
    special_config={

    },
    use_cuda=True,
    max_num=-1,
    seed=0,
    force_save=True,
    use_tqdm=False
)
# train_single_regression(
#     dataset=SingleRegressionDataset.FREESOLV,
#     data_name='FreeSolv-Xconf',
#     tag='FreeSolv-Xconf',
#     special_config={
#         'CONF_TYPE': ConfType.NONE,
#         'MESSAGE_TYPE': 'naive',
#     },
#     use_cuda=True,
#     max_num=-1,
#     seed=0,
#     force_save=False,
#     use_tqdm=False
# )
