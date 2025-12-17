import torch
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError
from sklearn.neighbors import kneighbors_graph
import torch_geometric
import itertools
import numpy as np


def minus(feature_1, feature_2):
    res = feature_1 - feature_2
    return res


def build_graphs_main_1(
    feature_matrix: torch.Tensor, n_neighbors: int, num_workers: int
):

    edge_index = torch_geometric.nn.knn_graph(
        feature_matrix,
        k=n_neighbors,
        batch=None,
        loop=False,
        flow="target_to_source",
        cosine=False,
        num_workers=num_workers,
    )
    edge_attr = []
    for ind in range(edge_index.shape[1]):
        edge_attr.append(
            minus(
                feature_matrix[edge_index[:, ind][0]],
                feature_matrix[edge_index[:, ind][1]],
            )
        )
    edge_attr = torch.stack(edge_attr)
    return edge_index, edge_attr


def calculate_edge_attr_1(
    edge_index: torch.Tensor,
    feature_matrix: torch.Tensor,
    n_neighbors: int,
    num_workers: int,
):

    edge_attr = []
    for ind in range(edge_index.shape[1]):
        edge_attr.append(
            minus(
                feature_matrix[edge_index[:, ind][0]],
                feature_matrix[edge_index[:, ind][1]],
            )
        )
    edge_attr = torch.stack(edge_attr)
    return edge_attr
