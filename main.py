import time
import torch
import rdkit.Chem as Chem
import numpy as np

from data.geom_qm9.load_qm9 import cache_qm9, load_qm9
from net.utils.model_utils import normalize_adj_r, normalize_adj_rc
from train.utils.loss_functions import distance_among

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    # cat
    # m1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    # m2 = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    # c1 = torch.cat([m1, m2])
    # c2 = torch.cat([m1, m2], dim=1)
    # print(c1.numpy())
    # print(c2.numpy())

    # diag
    # a = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
    # d = torch.diag(torch.reshape(a, [-1]))
    # print(d.numpy())

    # normalize
    # w = torch.tensor([[1, 1, 0], [0, 0, 1]], dtype=torch.float32)
    # norm_w = w / torch.sum(w, dim=-1, keepdim=True)
    # print(norm_w.numpy())

    # w1 w2
    # w1 = torch.tensor([[1, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.float32)
    # w2 = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    # adj_d = w1 @ w2.t()
    # adj = adj_d + adj_d.t()
    # print(adj)
    # adj_1 = normalize_adj_rc(adj)
    # print(adj_1)
    # print(adj_1 @ adj_1)
    # print(adj_1 @ adj_1 @ adj_1)
    # print(adj.diag())
    # print(adj.diagonal())

    # normalize_adj_rc
    # a1 = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32)
    # a2 = torch.tensor([[0, 0], [1, 2]], dtype=torch.float32)
    # na1 = normalize_adj_r(a1)
    # na2 = normalize_adj_r(a2)
    # print(na1.numpy())
    # print(na2.numpy())

    # utils geom_qm9
    # cache_qm9()

    # load geom_qm9
    # t0 = time.time()
    # m, n = load_qm9()
    # t1 = time.time()
    # print(t1 - t0)
    # print(m[0])
    # print(n[0])

    # np.mean
    # a = np.array([[1, 2], [3, 6], [6, 5]])
    # m = np.mean(a, axis=0)
    # sd = np.std(a, axis=0, ddof=1)
    # print(a)
    # print(a - m)
    # print(sd)
    # print((a - m) / sd)

    # distance_among
    a = torch.tensor([[1, 2], [3, 6], [6, 5]], dtype=torch.float32)
    d = distance_among(a)
    print(d)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
