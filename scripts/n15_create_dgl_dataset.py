import os
import json
import pandas as pd
import torch
import dgl
import numpy as np
from tqdm import tqdm

from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs, save_info, load_info


class GraphsFromCSVDataset(DGLDataset):
    def __init__(
        self,
        nodes_csv,
        edges_csv,
        metadata_json,
        hierarchy_root,
        raw_dir=None,
        save_dir=None,
        force_reload=False,
        verbose=False,
    ):
        self.nodes_csv = nodes_csv
        self.edges_csv = edges_csv
        self.metadata_json = metadata_json
        self.hierarchy_root = hierarchy_root
        with open(self.metadata_json, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        for path_part in self.hierarchy_root.split("\t"):
            if not path_part:
                break
            self.metadata = self.metadata[path_part]["subs"]

        self.meta = {}
        for class_type, class_info in self.metadata.items():
            for seq_id in class_info["sequences"]:
                self.meta[seq_id] = class_type

        super().__init__(
            name="graphs_from_csv",
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

    def process(self):
        nodes_df = pd.read_csv(
            self.nodes_csv,
            dtype={"graph_id": str, "node_id": np.int64},
        )
        edges_df = pd.read_csv(
            self.edges_csv,
            dtype={"graph_id": str, "src": np.int64, "dst": np.int64},
        )

        self.graphs = []
        self.graph_ids = []
        label_names = []

        node_feat_cols = [c for c in nodes_df.columns if c not in ["graph_id", "node_id"]]

        # группировка один раз вместо фильтрации DataFrame в цикле
        nodes_groups = {str(gid): g for gid, g in nodes_df.groupby("graph_id", sort=False)}
        edges_groups = {str(gid): g for gid, g in edges_df.groupby("graph_id", sort=False)}

        graph_ids = sorted(set(nodes_groups) & set(edges_groups) & set(self.meta))

        for graph_id in tqdm(graph_ids):
            g_nodes = nodes_groups[graph_id]
            g_edges = edges_groups[graph_id]

            if g_nodes.empty or g_edges.empty:
                continue

            # порядок узлов фиксируем по первому появлению, без сортировки
            unique_node_ids = pd.unique(g_nodes["node_id"])
            node_id_map = {nid: i for i, nid in enumerate(unique_node_ids)}

            src_raw = g_edges["src"].to_numpy()
            dst_raw = g_edges["dst"].to_numpy()

            missing_nodes = (set(src_raw) | set(dst_raw)) - set(unique_node_ids)
            if missing_nodes:
                raise ValueError(
                    f"В графе {graph_id} есть узлы в edges.csv, которых нет в nodes.csv: {missing_nodes}"
                )

            # исходные рёбра
            src = np.fromiter((node_id_map[x] for x in src_raw), dtype=np.int64, count=len(src_raw))
            dst = np.fromiter((node_id_map[x] for x in dst_raw), dtype=np.int64, count=len(dst_raw))

            # делаем bidirected вручную, без to_bidirected
            src_full = np.concatenate([src, dst])
            dst_full = np.concatenate([dst, src])

            g = dgl.graph((src_full, dst_full), num_nodes=len(unique_node_ids))

            # признаки узлов в том же порядке, что unique_node_ids
            g_nodes_indexed = (
                g_nodes.drop_duplicates("node_id", keep="first")
                .set_index("node_id")
                .loc[unique_node_ids]
                .reset_index()
            )

            node_feats = torch.from_numpy(
                g_nodes_indexed[node_feat_cols].to_numpy(dtype=np.float32, copy=False)
            )
            g.ndata["feat"] = node_feats
            g.ndata["node_id"] = torch.from_numpy(unique_node_ids.astype(np.int64, copy=False))

            if "edge_param" in g_edges.columns:
                edge_feat = torch.from_numpy(
                    g_edges["edge_param"].to_numpy(dtype=np.float32, copy=False)
                ).unsqueeze(1)

                # фичи для обратных рёбер дублируем в том же порядке
                edge_feat_full = torch.cat([edge_feat, edge_feat], dim=0)

                if g.num_edges() != edge_feat_full.shape[0]:
                    raise ValueError(
                        f"В графе {graph_id} число edge features ({edge_feat_full.shape[0]}) "
                        f"не совпадает с числом рёбер ({g.num_edges()})"
                    )

                g.edata["edge_param"] = edge_feat_full

            label_name = self.meta[graph_id]
            label_names.append(label_name)

            self.graphs.append(g)
            self.graph_ids.append(graph_id)

        self.label2id = {label: i for i, label in enumerate(sorted(set(label_names)))}
        self.id2label = {i: label for label, i in self.label2id.items()}

        self.labels = torch.tensor(
            [self.label2id[self.meta[graph_id]] for graph_id in self.graph_ids],
            dtype=torch.long,
        )

    def split_idx(
            self,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42,
            stratified=True,
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        np.random.seed(seed)

        indices = np.arange(len(self.graphs))

        if stratified:
            train_idx, val_idx, test_idx = [], [], []

            for label in torch.unique(self.labels):
                label = label.item()
                label_indices = indices[self.labels.numpy() == label]
                np.random.shuffle(label_indices)

                n = len(label_indices)
                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)

                train_idx.extend(label_indices[:n_train])
                val_idx.extend(label_indices[n_train:n_train + n_val])
                test_idx.extend(label_indices[n_train + n_val:])

        else:
            np.random.shuffle(indices)
            n = len(indices)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]

        return {'train': torch.tensor(train_idx, dtype=torch.long),
            'valid': torch.tensor(val_idx, dtype=torch.long),
            'test': torch.tensor(test_idx, dtype=torch.long),
        }


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
        nodes_csv=f"/Users/nad/hse/semester08/mobiraph/data/params_20_16_filter/nodes.csv",
        edges_csv=f"/Users/nad/hse/semester08/mobiraph/data/params_20_16_filter/edges.csv",
        metadata_json=f"/Users/nad/hse/semester08/mobiraph/data/n13_repbase_processed/hierarchy_sequences_02.json",
        hierarchy_root="",
        save_dir=f"{config.DIR_DOTPLOTS}/dgl_dataset/new",
        force_reload=True,
        verbose=True,
    )

    print("Число графов:", len(dataset))

    g, y = dataset[0]
    print(g)
    print("label id:", y.item())
    print("label name:", dataset.id2label[y.item()])