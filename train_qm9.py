from data.qm9.load_qm9 import cache_qm9
from train.train_qm9 import train_qm9

# cache_qm9()
# train_qm9(
#     special_config={
#         'Xconf': True,
#     },
#     use_cuda=True,
#     max_num=-1,
#     data_name='QM9',
#     seed=0,
#     force_save=True,
#     tag='QM9-Xconf',
#     use_tqdm=False,
# )

train_qm9(
    special_config={},
    use_cuda=True,
    max_num=-1,
    data_name='QM9',
    seed=0,
    force_save=False,
    tag='QM9-',
    use_tqdm=False,
)
