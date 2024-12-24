import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = nn.Linear(in_channels, hidden_channels)
        self.conv2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        edge_src, edge_dst = edge_index
        x_dst = x[edge_dst]

        x = self.conv1(x)  # Первый линейный слой
        x_dst = self.conv1(x_dst)

        # Агрегация признаков
        agg_x = torch.zeros_like(x)
        agg_x[edge_dst] += x_dst

        x = x + agg_x  # Суммируем признаки узлов
        x = self.conv2(x)  # Применяем второй линейный слой
        return x

