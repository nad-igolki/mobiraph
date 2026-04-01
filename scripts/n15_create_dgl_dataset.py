import os
import json
import pandas as pd
import torch
import dgl

from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs, save_info, load_info


class GraphsFromCSVDataset(DGLDataset):
    def __init__(
        self,
        nodes_csv,
        edges_csv,
        labels_json,
        raw_dir=None,
        save_dir=None,
        force_reload=False,
        verbose=False,
    ):
        self.nodes_csv = nodes_csv
        self.edges_csv = edges_csv
        self.labels_json = labels_json

        super().__init__(
            name="graphs_from_csv",
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

    def process(self):
        nodes_df = pd.read_csv(self.nodes_csv)
        edges_df = pd.read_csv(self.edges_csv)

        with open(self.labels_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.graphs = []
        self.graph_ids = []
        label_names = []

        node_feat_cols = [c for c in nodes_df.columns if c not in ["graph_id", "node_id"]]

        # берём только те graph_id, которые есть в nodes, edges и labels
        graph_ids = sorted(
            set(nodes_df["graph_id"])
            .intersection(set(edges_df["graph_id"]))
            .intersection(set(meta.keys()))
        )

        for graph_id in graph_ids:
            g_nodes = nodes_df[nodes_df["graph_id"] == graph_id].copy()
            g_edges = edges_df[edges_df["graph_id"] == graph_id].copy()

            if g_nodes.empty:
                continue
            if g_edges.empty:
                continue

            graph_label_name = meta[graph_id]["type"]
            label_names.append(graph_label_name)

            unique_node_ids = sorted(g_nodes["node_id"].unique())
            node_id_map = {node_id: i for i, node_id in enumerate(unique_node_ids)}

            edge_node_ids = set(g_edges["src"]).union(set(g_edges["dst"]))
            missing_nodes = edge_node_ids - set(unique_node_ids)
            if missing_nodes:
                raise ValueError(
                    f"В графе {graph_id} есть узлы в edges.csv, которых нет в nodes.csv: {missing_nodes}"
                )

            src = g_edges["src"].map(node_id_map).to_list()
            dst = g_edges["dst"].map(node_id_map).to_list()

            g = dgl.graph((src, dst), num_nodes=len(unique_node_ids))

            g_nodes = g_nodes.sort_values("node_id")
            node_feats = torch.tensor(g_nodes[node_feat_cols].values, dtype=torch.float32)
            g.ndata["feat"] = node_feats
            g.ndata["node_id"] = torch.tensor(g_nodes["node_id"].values, dtype=torch.int64)

            if "edge_param" in g_edges.columns:
                edge_feat = torch.tensor(
                    g_edges["edge_param"].values, dtype=torch.float32
                ).unsqueeze(1)
                g.edata["edge_param"] = edge_feat

            self.graphs.append(g)
            self.graph_ids.append(graph_id)

        # кодируем строковые labels в числа
        self.label2id = {label: i for i, label in enumerate(sorted(set(label_names)))}
        self.id2label = {i: label for label, i in self.label2id.items()}

        self.labels = []
        for graph_id in self.graph_ids:
            label_name = meta[graph_id]["type"]
            self.labels.append(self.label2id[label_name])

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    def save(self):
        graph_path = os.path.join(self.save_path, "graphs.bin")
        info_path = os.path.join(self.save_path, "info.pkl")

        save_graphs(graph_path, self.graphs)
        save_info(
            info_path,
            {
                "graph_ids": self.graph_ids,
                "labels": self.labels.tolist(),
                "label2id": self.label2id,
                "id2label": self.id2label,
            },
        )

    def load(self):
        graph_path = os.path.join(self.save_path, "graphs.bin")
        info_path = os.path.join(self.save_path, "info.pkl")

        self.graphs, _ = load_graphs(graph_path)
        info = load_info(info_path)

        self.graph_ids = info["graph_ids"]
        self.labels = torch.tensor(info["labels"], dtype=torch.long)
        self.label2id = info["label2id"]
        self.id2label = {int(k): v for k, v in info["id2label"].items()} if isinstance(next(iter(info["id2label"].keys())), str) else info["id2label"]

    def has_cache(self):
        graph_path = os.path.join(self.save_path, "graphs.bin")
        info_path = os.path.join(self.save_path, "info.pkl")
        return os.path.exists(graph_path) and os.path.exists(info_path)

if __name__ == "__main__":
    import config
    dataset = GraphsFromCSVDataset(
        nodes_csv=f"{config.DIR_DOTPLOTS}/nodes.csv",
        edges_csv=f"{config.DIR_DOTPLOTS}/edges.csv",
        labels_json=f"{config.DIR_REPBASE_PROCESSED}/metadata.json",
        save_dir=f"{config.DIR_DOTPLOTS}/dgl_dataset",
        force_reload=True,
        verbose=True,
    )

    print("Число графов:", len(dataset))

    g, y = dataset[0]
    print(g)
    print("label id:", y.item())
    print("label name:", dataset.id2label[y.item()])