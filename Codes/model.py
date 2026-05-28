#!/usr/bin/env python3
"""
AFP Prediction Model - V4 Architecture
GCN-BiLSTM-Transformer
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


class AFPTransformerGNN(nn.Module):
    """GCN-BiLSTM-Transformer Antifungal Peptide Prediction Model"""

    def __init__(self, input_dim=26, hidden_dim=128, num_heads=8,
                 num_gat_layers=3, num_classes=2, dropout=0.3):
        super().__init__()
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
            self.gat_layers.append(
                GATConv(in_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True)
            )
        self.gat_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * num_heads) for _ in range(num_gat_layers)
        ])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * num_heads, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_heads * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_embedding(x)
        for gat, norm in zip(self.gat_layers, self.gat_norms):
            x = torch.relu(norm(gat(x, edge_index)))
            x = self.dropout(x)
        x_padded, mask = self._pad_sequence(x, batch)
        x_transformed = self.transformer(x_padded, src_key_padding_mask=~mask)[mask]
        x_pool = global_mean_pool(x_transformed, batch)
        x_max = self._global_max_pool(x_transformed, batch)
        return self.classifier(torch.cat([x_pool, x_max], dim=1))

    def _pad_sequence(self, x, batch):
        batch_size = batch.max().item() + 1
        max_len = max((batch == i).sum().item() for i in range(batch_size))
        padded = torch.zeros(batch_size, max_len, x.size(1), device=x.device)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=x.device)
        idx = 0
        for i in range(batch_size):
            seq_len = (batch == i).sum().item()
            padded[i, :seq_len] = x[idx:idx + seq_len]
            mask[i, :seq_len] = True
            idx += seq_len
        return padded, mask

    def _global_max_pool(self, x, batch):
        batch_size = batch.max().item() + 1
        out = torch.zeros(batch_size, x.size(1), device=x.device)
        for i in range(batch_size):
            mask = batch == i
            if mask.any():
                out[i] = x[mask].max(dim=0)[0]
        return out
