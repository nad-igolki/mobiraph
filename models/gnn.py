import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.nn import BatchNorm



class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GNN, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # Первый слой GATConv
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Второй слой GATConv
        x = self.conv2(x, edge_index)

        # Глобальная агрегация узловых признаков в графовые
        x = global_mean_pool(x, batch)

        # Финальный линейный слой для предсказания
        x = self.lin(x)

        return x



class ComplexGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, add_bn=True, dropout=0.5, heads=2):
        super(ComplexGAT, self).__init__()

        self.add_bn = add_bn
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if add_bn else None

        # Первый слой GATConv
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        if add_bn:
            self.bns.append(BatchNorm(hidden_channels * heads))

        # Остальные скрытые слои GATConv
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            if add_bn:
                self.bns.append(BatchNorm(hidden_channels * heads))

        # Последний слой GATConv (heads=1 для получения итогового представления)
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout))

        # Линейный слой
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.add_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Последний слой GATConv
        x = self.convs[-1](x, edge_index)

        # Глобальная агрегация узловых признаков
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # Финальный линейный слой
        out = self.lin(x)  # [batch_size, out_channels]

        return out
