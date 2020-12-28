import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from sklearn.metrics import roc_auc_score

from net.utils.MaskMatrices import MaskMatrices
from net.utils.model_utils import normalize_adj_rc


def multi_roc(source: np.ndarray, target: np.ndarray) -> Tuple[float, List[float]]:
    assert source.shape == target.shape
    not_nan_mask = np.logical_not(np.isnan(target))
    list_roc = []
    n_m = source.shape[1]
    for i in range(n_m):
        try:
            roc = roc_auc_score(target[not_nan_mask[:, i], i], source[not_nan_mask[:, i], i])
        except ValueError:
            roc = 1
        list_roc.append(roc)
    return sum(list_roc) / len(list_roc), list_roc


def multi_mse_loss(source: torch.Tensor, target: torch.Tensor, explicit=False) -> torch.Tensor:
    se = (source - target) ** 2
    mse = torch.mean(se, dim=0)
    if explicit:
        return mse
    else:
        return torch.sum(mse)


def multi_mae_loss(source: torch.Tensor, target: torch.Tensor, explicit=False) -> torch.Tensor:
    ae = torch.abs(source - target)
    mae = torch.mean(ae, dim=0)
    if explicit:
        return mae
    else:
        return torch.sum(mae)


def mse_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(source, target)


def rmse_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(source, target).sqrt()


def mae_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(source - target))


def distance_among(positions: torch.Tensor) -> torch.Tensor:
    p1 = torch.unsqueeze(positions, 0)
    p2 = torch.unsqueeze(positions, 1)
    distance = torch.norm(p1 - p2, dim=2)
    return distance


def adj3_loss(source: torch.Tensor, target: torch.Tensor, mask_matrices: MaskMatrices,
              use_cuda=False) -> torch.Tensor:
    n_atom = mask_matrices.mol_vertex_w.shape[1]
    vew1 = mask_matrices.vertex_edge_w1
    vew2 = mask_matrices.vertex_edge_w2
    adj_d = vew1 @ vew2.t()
    i = torch.eye(adj_d.shape[0])
    if use_cuda:
        i = i.cuda()
    adj = adj_d + adj_d.t() + i
    norm_adj = normalize_adj_rc(adj)
    norm_adj_2 = norm_adj @ norm_adj
    norm_adj_3 = norm_adj_2 @ norm_adj
    mean_adj_3 = (norm_adj + norm_adj_2 + norm_adj_3) / 3

    ds = distance_among(source)
    dt = distance_among(target)
    distance_2 = (ds - dt) ** 2
    loss = torch.sum(distance_2 * mean_adj_3) / n_atom
    return loss


def distance_loss(source: torch.Tensor, target: torch.Tensor, mask_matrices: MaskMatrices,
                  root_square=True) -> torch.Tensor:
    n_mol = mask_matrices.mol_vertex_w.shape[0]
    mvw = mask_matrices.mol_vertex_w
    vv = mvw.t() @ mvw
    norm_vv = vv / ((torch.sum(vv, dim=1) ** 2) * n_mol)
    ds = distance_among(source)
    dt = distance_among(target)
    if root_square:
        return torch.sqrt(torch.sum(((ds - dt) ** 2) * norm_vv))
    else:
        return torch.sum((ds - dt) * norm_vv)
