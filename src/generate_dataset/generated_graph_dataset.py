import os
import pandas as pd
import torch
import dgl
from dgl.data import DGLDataset
from sklearn.model_selection import train_test_split


class GeneratedGraphDataset(DGLDataset):
    def __init__(self, raw_dir='generate_dataset/generated_graphs', split_ratio=(0.8, 0.1, 0.1)):
        self.split_ratio = split_ratio
        super().__init__(name='generated_graph_dataset', raw_dir=raw_dir)

    def process(self):
        edges_df = pd.read_csv(os.path.join(self.raw_dir, 'edges.csv'))
        nodes_df = pd.read_csv(os.path.join(self.raw_dir, 'nodes.csv'))
        labels_df = pd.read_csv(os.path.join(self.raw_dir, 'graph_labels.csv'))

        graph_ids = labels_df['graph_id'].unique()
        self.graphs = []
        self.labels = []

        for gid in graph_ids:
            edge_data = edges_df[edges_df['graph_id'] == gid]
            node_data = nodes_df[nodes_df['graph_id'] == gid].sort_values(by='node_id')

            # Создание графа
            g = dgl.graph((edge_data['src'].values, edge_data['dst'].values))

            # Добавляем фичи узлов
            feat_columns = [col for col in node_data.columns if col.startswith('feat_')]
            feats = torch.tensor(node_data[feat_columns].values, dtype=torch.float32)
            g.ndata['feat'] = feats

            # Добавляем фичи ребер
            edge_feat_cols = [col for col in edge_data.columns if col not in ['graph_id', 'src', 'dst']]
            edge_features = torch.tensor(edge_data[edge_feat_cols].values, dtype=torch.float32)
            g.edata['feat'] = edge_features

            self.graphs.append(g)
            label = labels_df[labels_df['graph_id'] == gid]['label'].values[0]
            self.labels.append(label)

        self.labels = torch.tensor(self.labels)

        total_indices = list(range(len(self.graphs)))
        train_ratio, val_ratio, test_ratio = self.split_ratio

        train_ids, temp_ids = train_test_split(
            total_indices, train_size=train_ratio, shuffle=True
        )
        val_ids, test_ids = train_test_split(
            temp_ids, test_size=test_ratio / (val_ratio + test_ratio)
        )

        self.split_idx = {
            'train': train_ids,
            'valid': val_ids,
            'test': test_ids
        }

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)