import numpy as np
import matplotlib.pyplot as plt
import rdkit
import rdkit.Chem as Chem
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple

ATOM_CONFIG = {
    'C': (6, 'black', 's'),
    'N': (7, 'black', 'p'),
    'O': (8, 'black', 'o'),
    'F': (9, 'black', 'v'),
    'S': (16, 'black', 'h'),
}

DEFAULT_CONFIG = (10, 'black', '.')


def atom_config(atom: str) -> tuple:
    if atom in ATOM_CONFIG.keys():
        return ATOM_CONFIG[atom]
    print('## Undefined atom type {} ##'.format(atom))
    return DEFAULT_CONFIG


def get_atoms_size_color_marker(smiles: str) -> Tuple[list, list, list]:
    sizes = []
    colors = []
    markers = []
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom = atom.GetSymbol()
        size, color, marker = atom_config(atom)
        sizes.append(size * 20)
        colors.append(color)
        markers.append(marker)
    return sizes, colors, markers


def get_bonds_u_v_width_style(smiles: str) -> Tuple[list, list, list, list]:
    us = []
    vs = []
    widths = []
    styles = []
    mol = Chem.MolFromSmiles(smiles)
    for bond in mol.GetBonds():
        us.append(bond.GetBeginAtomIdx())
        vs.append(bond.GetEndAtomIdx())
        bond_type = bond.GetBondType()
        width = 5
        if bond_type == Chem.rdchem.BondType.SINGLE:
            style = '-'
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            style = '--'
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            style = ':'
        else:
            style = '-.'
        widths.append(width)
        styles.append(style)
    return us, vs, widths, styles


def plt_local_alignment(pos: np.ndarray, smiles: str, local_alignment: np.ndarray,
                        title: str = 'plt_3d', d: str = 'visualize/alignment/graph'):
    n_edge = int(local_alignment.shape[1] / 2)
    local_alignment = local_alignment * np.sum(local_alignment > 1e-5, axis=1, keepdims=True)
    edge_weight = np.sum(local_alignment, axis=0)
    edge_weight = (edge_weight[:n_edge] + edge_weight[n_edge]) / 2
    edge_weight = edge_weight - np.min(edge_weight) + 0.02
    edge_weight = edge_weight / (np.max(edge_weight) + 1e-5)

    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)
    # ax = fig.add_subplot(111, projection='3d')

    us, vs, linewidths, linestyles = get_bonds_u_v_width_style(smiles)
    for u, v, linewidth, linestyle, weight in zip(us, vs, linewidths, linestyles, edge_weight):
        ax.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]], [pos[u, 2], pos[v, 2]],
                linewidth=linewidth, linestyle=linestyle, c=[1 - weight, 1 - weight, 1])

    sizes, colors, markers = get_atoms_size_color_marker(smiles)
    for i in range(pos.shape[0]):
        ax.scatter(pos[i:i + 1, 0], pos[i:i + 1, 1], pos[i:i + 1, 2], s=sizes[i], c=colors[i], marker=markers[i])

    plt.title(title)
    plt.savefig('{}/{}.png'.format(d, title))
    plt.close()


def plt_global_alignment(pos: np.ndarray, smiles: str, global_alignment: np.ndarray,
                         title: str = 'plt_3d', d: str = 'visualize/alignment'):
    global_alignment = global_alignment[0]
    colors = global_alignment / np.max(global_alignment)

    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)
    # ax = fig.add_subplot(111, projection='3d')

    us, vs, linewidths, linestyles = get_bonds_u_v_width_style(smiles)
    for u, v, linewidth, linestyle in zip(us, vs, linewidths, linestyles):
        ax.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]], [pos[u, 2], pos[v, 2]],
                linewidth=linewidth, linestyle=linestyle, c='k')

    sizes, _, markers = get_atoms_size_color_marker(smiles)
    for i in range(pos.shape[0]):
        ax.scatter(pos[i:i + 1, 0], pos[i:i + 1, 1], pos[i:i + 1, 2],
                   s=sizes[i], c=[[1 - colors[i], 1 - colors[i], 1]], marker=markers[i])

    plt.title(title)
    plt.savefig('{}/{}.png'.format(d, title))
    plt.close()
