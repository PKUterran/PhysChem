from data.qm9.load_qm9 import cache_qm9
from train.train_qm9 import train_qm9

# cache_qm9()
train_qm9(max_num=5000, data_name='QM9-5000',
          use_tqdm=False, force_save=False, use_cuda=True,
          special_config={
              'HV_DIM': 16,
              'HE_DIM': 16,
              'HM_DIM': 16,
              'MV_DIM': 16,
              'ME_DIM': 16,
              'MM_DIM': 16,
              'PQ_DIM': 16,
              'N_LAYER': 4,
              'N_HOP': 3,
              'N_ITERATION': 10,
              'N_GLOBAL': 3,
              'MESSAGE_TYPE': 'triplet',
              'UNION_TYPE': 'gru',
              'DERIVATION_TYPE': 'newton',
              'TAU': 0.02,
              'DROPOUT': 0.0,

              'EPOCH': 100,
              'BATCH': 16,
              'LAMBDA': 1,
              'LR': 4e-4,
              'DECAY': 1e-5,
          })
