#!/usr/bin/env python3
"""
Feature Extraction Module
"""

import numpy as np
import torch
from torch_geometric.data import Data


AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_NUM = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
AA_TO_NUM['X'] = 20

Z_SCALE = {
    'A': [0.24, -2.32, 0.60, -0.14, 1.30],
    'C': [0.84, -1.67, 3.71, 0.18, -2.65],
    'D': [3.98, 0.93, 1.93, -2.46, 0.75],
    'E': [3.11, 0.26, -0.11, -3.04, -0.25],
    'F': [-4.22, 1.94, 1.06, 0.54, -0.62],
    'G': [2.05, -4.06, 0.36, -0.82, -0.38],
    'H': [2.47, 1.95, 0.26, 3.90, 0.09],
    'I': [-3.89, -1.73, -1.71, -0.84, 0.26],
    'K': [2.29, 0.89, -2.49, 1.49, 0.31],
    'L': [-4.28, -1.30, -1.49, -0.72, 0.84],
    'M': [-2.85, -0.22, 0.47, 1.94, -0.98],
    'N': [3.05, 1.62, 1.04, -1.15, 1.61],
    'P': [-1.66, 0.27, 1.84, 0.70, 2.00],
    'Q': [1.75, 0.50, -1.44, -1.34, 0.66],
    'R': [3.52, 2.50, -3.50, 1.99, -0.17],
    'S': [2.39, -1.07, 1.15, -1.39, 0.67],
    'T': [0.75, -2.18, -1.12, -1.46, -0.40],
    'V': [-2.59, -2.64, -1.54, -0.85, -0.02],
    'W': [-4.36, 3.94, 0.59, 3.44, -1.59],
    'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],
    'X': [0.0, 0.0, 0.0, 0.0, 0.0],
}


def extract_features(seq):
    """Extract sequence features (zscale 5D + onehot 21D = 26D)"""
    features = []
    for aa in seq.upper():
        zscale = Z_SCALE.get(aa, [0.0] * 5)
        onehot = [0.0] * 21
        onehot[AA_TO_NUM.get(aa, 20)] = 1.0
        features.append(zscale + onehot)
    return np.array(features, dtype=np.float32)


def build_graph(seq_features):
    """Build sequence graph structure"""
    num_nodes = len(seq_features)
    edge_index = []
    for i in range(num_nodes - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()


def sequence_to_data(sequence):
    """Convert sequence to PyG Data object"""
    features = extract_features(sequence)
    edge_index = build_graph(features)
    x = torch.tensor(features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    return data
