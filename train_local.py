# from net.config import ConfType
# from train.train_qm9 import train_qm9
#
# train_qm9(
#     special_config={
#         'INIT_GCN_H_DIMS': [128],
#         'INIT_GCN_O_DIM': 128,
#         'INIT_LSTM_LAYERS': 4,
#         'INIT_LSTM_O_DIM': 128,
#         'HV_DIM': 128,
#         'HE_DIM': 128,
#         'HM_DIM': 128,
#         'MV_DIM': 128,
#         'ME_DIM': 128,
#         'MM_DIM': 128,
#         'PQ_DIM': 3,
#         'N_LAYER': 2,
#         'N_HOP': 1,
#         'N_ITERATION': 4,
#         'N_GLOBAL': 2,
#         'TAU': 0.05,
#         'DISSA': 1.0,
#         'EPOCH': 200,
#         'BATCH': 32,
#         'LAMBDA': 100,
#         'LR': 1e-4,
#         'GAMMA': 0.99,
#         'DECAY': 1e-5,
#         # 'CONF_TYPE': ConfType.RDKIT,
#     },
#     use_cuda=True,
#     max_num=3400,
#     data_name='QM9-3400',
#     seed=0,
#     force_save=True,
#     tag='QM9-3400-init3',
#     use_tqdm=False,
# )
from net.config import ConfType
from train.train_single_regression import train_single_regression, SingleRegressionDataset
seed = 16880611
pos = 1

extra_config = {}
if pos == 0:
    name = 'ESOL-Xconf'
    conf_type = ConfType.NONE
elif pos == 1:
    name = 'ESOL'
    conf_type = ConfType.RDKIT
else:
    name = 'ESOL-RGT'
    conf_type = ConfType.NEWTON_RGT
    extra_config = {}

train_single_regression(
    dataset=SingleRegressionDataset.ESOL,
    data_name=f'{name}@{seed}',
    tag=f'{name}@{seed}-test',
    special_config=dict({
        'CONF_TYPE': conf_type,
        'DROPOUT': 0,
    }, **extra_config),
    use_cuda=False,
    max_num=-1,
    seed=seed,
    force_save=True,
    use_tqdm=False
)

