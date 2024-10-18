import argparse
import os.path as osp
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GINEConv, GPSConv,GATConv,GatedGraphConv
from torch_geometric.nn.attention import PerformerAttention



class GPS(nn.Module):
    def __init__(self, channels: int, pe_dim: int, num_layers: int,
                 attn_type: str, attn_kwargs: Dict[str, Any]):
        super().__init__()

        self.node_emb = nn.Embedding(44, channels-pe_dim)#28
        self.pe_lin = nn.Linear(8, pe_dim)#20
        # self.pe_lin = nn.Embedding(8, pe_dim)#2
        self.pe_norm = nn.BatchNorm1d(8)#20
        self.edge_emb = nn.Embedding(10, channels)#4
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            ss = nn.Sequential(
                nn.Linear(channels, channels),
                nn.ReLU(),
                nn.Linear(channels, channels),
            )
            conv = GPSConv(channels, GINEConv(ss), heads=4,
                           attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)
        self.in_feat_dropout = nn.Dropout(0.0)
        self.embedding_lap_pos_enc = nn.Linear(8, 128)
        self.linear_h = nn.Linear(44, 128)
        self.linear_e = nn.Linear(10, 128)#10,128
    def forward(self, x, pe, edge_index, edge_attr, batch):
        h = self.linear_h(x.float())
        h_lap_pos_enc = self.embedding_lap_pos_enc(pe.float())
        h = h + h_lap_pos_enc
        x = self.in_feat_dropout(h)
        edge_attr = self.linear_e(edge_attr.float())

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        return x


class RedrawProjection:
    def __init__(self, model: nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1


