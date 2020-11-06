import torch
import numpy as np

DEFAULT_SEEDS = [

]


def set_seed(seed: int, use_cuda=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
