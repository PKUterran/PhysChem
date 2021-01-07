import numpy as np
import matplotlib.pyplot as plt
import rdkit
import rdkit.Chem as Chem
from mpl_toolkits.mplot3d import Axes3D


ATOM_CONFIG = {
    'C': (6, 'brown', 's'),
    'N': (7, 'blue', 'p'),
    'O': (8, 'red', 'o'),
    'F': (9, 'purple', 'v'),
    'S': (16, 'yellow', 'h'),
}

DEFAULT_CONFIG = (10, 'black', '.')


def atom_config(atom: str) -> tuple:
    if atom in ATOM_CONFIG.keys():
        return ATOM_CONFIG[atom]
    print('## Undefined atom type {} ##'.format(atom))
    return DEFAULT_CONFIG


def get_atoms_size_color_marker(smiles: str) -> (list, list, list):
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


def get_bonds_u_v_width_style(smiles) -> (list, list, list):
    us = []
    vs = []
    widths = []
    styles = []
    mol = Chem.MolFromSmiles(smiles)
    for bond in mol.GetBonds():
        us.append(bond.GetBeginAtomIdx())
        vs.append(bond.GetEndAtomIdx())
        bond_type = bond.GetBondType()
        if bond_type == Chem.rdchem.BondType.SINGLE:
            width = 2
            style = '-'
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            width = 4
            style = '--'
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            width = 6
            style = ':'
        else:
            width = 3
            style = '-.'
        widths.append(width)
        styles.append(style)
    return us, vs, widths, styles


def plt_molecule_3d(pos: np.ndarray, smiles: str, title: str = 'plt_3d', d: str = 'visualize/graph'):
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)
    # ax = fig.add_subplot(111, projection='3d')

    us, vs, linewidths, linestyles = get_bonds_u_v_width_style(smiles)
    for u, v, linewidth, linestyle in zip(us, vs, linewidths, linestyles):
        ax.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]], [pos[u, 2], pos[v, 2]],
                linewidth=linewidth, linestyle=linestyle, c='k')

    sizes, colors, markers = get_atoms_size_color_marker(smiles)
    for i in range(pos.shape[0]):
        ax.scatter(pos[i:i+1, 0], pos[i:i+1, 1], pos[i:i+1, 2], s=sizes[i], c=colors[i], marker=markers[i])

    plt.title(title)
    plt.savefig('{}/{}.png'.format(d, title))
    plt.close()
