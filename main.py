import torch

from net.utils.model_utils import normalize_adj_r

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

    # normalize_adj_rc
    a1 = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32)
    a2 = torch.tensor([[0, 0], [1, 2]], dtype=torch.float32)
    na1 = normalize_adj_r(a1)
    na2 = normalize_adj_r(a2)
    print(na1.numpy())
    print(na2.numpy())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
