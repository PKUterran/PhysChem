import torch
import numpy as np

DEFAULT_SEEDS = [
    16880611,
    17760704,
    17890714,
    19491001,
    19900612,
]


def set_seed(seed: int, use_cuda=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
