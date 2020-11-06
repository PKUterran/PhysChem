import torch


class MaskMatrices:
    def __init__(self, mol_vertex_w: torch.Tensor, mol_vertex_b: torch.Tensor,
                 vertex_edge_w1: torch.Tensor, vertex_edge_w2: torch.Tensor,
                 vertex_edge_b1: torch.Tensor, vertex_edge_b2: torch.Tensor):
        self.mol_vertex_w = mol_vertex_w
        self.mol_vertex_b = mol_vertex_b
        self.vertex_edge_w1 = vertex_edge_w1
        self.vertex_edge_w2 = vertex_edge_w2
        self.vertex_edge_b1 = vertex_edge_b1
        self.vertex_edge_b2 = vertex_edge_b2
