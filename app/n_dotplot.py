from scripts.n14_create_dotplot_embeddings import create_and_write_graphs_from_fasta_parallel
from pathlib import Path
from scripts.n15_create_dgl_dataset import GraphsFromCSVDataset
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from torch.utils.data import DataLoader, Subset
from dgl.nn import GATv2Conv, AvgPooling, MaxPooling


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(samples):
    graph_ids, graphs, labels = map(list, zip(*samples))
    bg = dgl.batch(graphs)
    labels = torch.tensor(labels)
    return graph_ids, bg, labels

import torch
import pandas as pd

@torch.no_grad()
def predict_with_logits(model, dataloader, device):
    model.eval()

    all_graph_ids = []
    all_logits = []
    all_preds = []
    all_labels = []

    for graph_ids, bg, labels in dataloader:
        bg = bg.to(device)

        logits = model(bg)
        preds = logits.argmax(dim=1).cpu()

        all_graph_ids.extend(graph_ids)
        all_logits.append(logits.cpu())
        all_preds.append(preds)
        all_labels.append(labels)

    return (
        all_graph_ids,
        torch.cat(all_logits, dim=0),
        torch.cat(all_preds, dim=0),
        torch.cat(all_labels, dim=0),
    )


class GATGraphClassifier(nn.Module):
    def __init__(
        self,
        in_feats,
        hidden_feats,
        num_classes,
        num_heads=4,
        dropout=0.2,
        use_max_pool=True,
    ):
        super().__init__()

        self.gat1 = GATv2Conv(
            in_feats=in_feats,
            out_feats=hidden_feats,
            num_heads=num_heads,
            feat_drop=dropout,
            attn_drop=dropout,
            allow_zero_in_degree=True,
        )

        self.gat2 = GATv2Conv(
            in_feats=hidden_feats,
            out_feats=hidden_feats,
            num_heads=num_heads,
            feat_drop=dropout,
            attn_drop=dropout,
            allow_zero_in_degree=True,
        )

        self.avg_pool = AvgPooling()
        self.max_pool = MaxPooling() if use_max_pool else None

        readout_dim = hidden_feats * 2 if use_max_pool else hidden_feats

        self.classifier = nn.Sequential(
            nn.Linear(readout_dim, hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, num_classes),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, g):
        x = g.ndata["feat"].float()

        # [N, num_heads, hidden_feats]
        x = self.gat1(g, x)
        x = x.mean(dim=1)  # -> [N, hidden_feats]
        x = F.relu(x)
        x = self.dropout(x)

        # [N, num_heads, hidden_feats]
        x = self.gat2(g, x)
        x = x.mean(dim=1)  # -> [N, hidden_feats]
        x = F.relu(x)

        g.ndata["h"] = x

        h_avg = self.avg_pool(g, x)

        if self.max_pool is not None:
            h_max = self.max_pool(g, x)
            hg = torch.cat([h_avg, h_max], dim=1)
        else:
            hg = h_avg

        logits = self.classifier(hg)
        return logits


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    for graph_ids, bg, labels in dataloader:
        bg = bg.to(device)
        labels = labels.to(device)

        logits = model(bg)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc



def train_model(
    dataset,
    split_idx,
    hidden_feats=64,
    num_heads=4,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    num_epochs=50,
    dropout=0.2,
    device=None,
    use_max_pool=True,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = Subset(dataset, split_idx["train"].tolist())
    val_dataset = Subset(dataset, split_idx["valid"].tolist())
    test_dataset = Subset(dataset, split_idx["test"].tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    _, sample_graph, _ = dataset[0]
    in_feats = sample_graph.ndata["feat"].shape[1]
    num_classes = len(dataset.label2id)

    model = GATGraphClassifier(
        in_feats=in_feats,
        hidden_feats=hidden_feats,
        num_classes=num_classes,
        num_heads=num_heads,
        dropout=dropout,
        use_max_pool=use_max_pool,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for graph_ids, bg, labels in train_loader:
            bg = bg.to(device)
            labels = labels.to(device)

            logits = model(bg)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        val_loss, val_acc = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\nBest val_acc={best_val_acc:.4f}")
    print(f"Test loss={test_loss:.4f}, test_acc={test_acc:.4f}")

    return model

def dotplot_train(fasta_path: str, checkpoint_dir: str, metadata_file: str, wsize = 20, nmatch = 16, processes: int | None = None):
    dotplots_dir = Path(checkpoint_dir) / "dotplot" / "edges_nodes"
    dotplots_dir.mkdir(parents=True, exist_ok=True)
    dgl_dataset_dir = Path(checkpoint_dir) / "dotplot" / "dgl_dataset"
    dgl_dataset_dir.mkdir(parents=True, exist_ok=True)
    create_and_write_graphs_from_fasta_parallel(
        fasta_path=fasta_path,
        output_dir=dotplots_dir,
        wsize=wsize,
        nmatch=nmatch,
        scatter=False,
        n_processes=processes,
        reset_outputs=False,
    )
    HIERARCHY_ROOTS = [
        "",
        "Class I (Retrotransposons)",
        "Class II (DNA transposons)",
        "Class I (Retrotransposons)\tLTR Retrotransposon",
        "Class I (Retrotransposons)\tNon-LTR Retrotransposon",
    ]

    for hierarchy_root in HIERARCHY_ROOTS:
        dataset = GraphsFromCSVDataset(
            nodes_csv=f"{str(dotplots_dir)}/nodes.csv",
            edges_csv=f"{str(dotplots_dir)}/edges.csv",
            metadata_json=metadata_file,
            hierarchy_root=hierarchy_root,
            save_dir=str(dgl_dataset_dir),
            force_reload=True,
            verbose=True,
        )

        print("Уровень иерархии:", hierarchy_root)
        print("Число графов:", len(dataset))
        print("Классы:", dataset.label2id)

        split_idx = dataset.split_idx(
            train_ratio=0.9,
            val_ratio=0.05,
            test_ratio=0.05,
            seed=42,
            stratified=True,
        )

        model = train_model(
            dataset=dataset,
            split_idx=split_idx,
            hidden_feats=64,
            batch_size=32,
            lr=1e-3,
            weight_decay=1e-4,
            num_epochs=10,
            dropout=0.2,
            use_max_pool=False
        )
        model_dir = Path(checkpoint_dir) / "dotplot" / "models"
        if hierarchy_root == "":
            model_dir = model_dir / "root"
        else:
            model_dir = model_dir / hierarchy_root

        model_dir.mkdir(parents=True, exist_ok=True)

        save_path = model_dir / "gat_model.pt"
        torch.save(model.state_dict(), save_path)

        print(f"Model saved to {save_path}")


        id2label = {v: k for k, v in dataset.label2id.items()}

        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=collate_fn,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        graph_ids, logits, preds, labels = predict_with_logits(model, loader, device)

        df_logits = pd.DataFrame(
            logits.numpy(),
            columns=[f"logit_{i}" for i in range(logits.shape[1])]
        )

        df_logits.insert(0, "graph_id", graph_ids)

        df_logits["pred_class_id"] = preds.numpy()
        df_logits["true_class_id"] = labels.numpy()
        df_logits["y_pred"] = df_logits["pred_class_id"].map(id2label)
        df_logits["y_true"] = df_logits["true_class_id"].map(id2label)

        df_logits.drop(columns=["pred_class_id", "true_class_id"], inplace=True)

        cols = ["graph_id"] + [c for c in df_logits.columns if c != "graph_id"]
        df_logits = df_logits[cols]

        df_logits.rename(columns={'graph_id': 'name'}, inplace=True)
        df_logits.drop(columns=["y_true"], inplace=True)

        logits_dir = Path(checkpoint_dir) / "dotplot" / "logits"
        if hierarchy_root == "":
            df_logits.to_csv(logits_dir / "root" / "gat.csv", index=False)
        else:
            df_logits.to_csv(logits_dir / hierarchy_root / "gat.csv", index=False)

        print("Сохранено в gat.csv")
        df_logits.head()


def dotplot_test(
    fasta_path: str,
    checkpoint_dir: str,
    models_path: str,
    wsize=20,
    nmatch=16,
    processes: int | None = None,
    batch_size: int = 32
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dotplots_dir = Path(checkpoint_dir) / "dotplot" / "edges_nodes"
    dotplots_dir.mkdir(parents=True, exist_ok=True)

    dgl_dataset_dir = Path(checkpoint_dir) / "dotplot" / "dgl_dataset"
    dgl_dataset_dir.mkdir(parents=True, exist_ok=True)

    logits_dir = Path(checkpoint_dir) / "dotplot" / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)

    create_and_write_graphs_from_fasta_parallel(
        fasta_path=fasta_path,
        output_dir=dotplots_dir,
        wsize=wsize,
        nmatch=nmatch,
        scatter=False,
        n_processes=processes,
        reset_outputs=False,
    )

    HIERARCHY_ROOTS = [
        "",
        "Class I (Retrotransposons)",
        "Class II (DNA transposons)",
        "Class I (Retrotransposons)\tLTR Retrotransposon",
        "Class I (Retrotransposons)\tNon-LTR Retrotransposon",
    ]

    for hierarchy_root in HIERARCHY_ROOTS:
        dataset = GraphsFromCSVDataset(
            nodes_csv=f"{str(dotplots_dir)}/nodes.csv",
            edges_csv=f"{str(dotplots_dir)}/edges.csv",
            metadata_json=None,
            hierarchy_root=hierarchy_root,
            save_dir=str(dgl_dataset_dir),
            force_reload=True,
            verbose=True,
        )

        print("Уровень иерархии:", hierarchy_root)
        print("Число графов:", len(dataset))
        print("Классы:", dataset.label2id)

        sample_graph, _ = dataset[0]
        in_feats = sample_graph.ndata["feat"].shape[1]
        num_classes = len(dataset.label2id)

        model = GATGraphClassifier(
            in_feats=in_feats,
            hidden_feats=64,
            num_classes=num_classes,
            num_heads=4,
            dropout=0.2,
            use_max_pool=False,
        ).to(device)

        current_model_dir = Path(models_path)
        if hierarchy_root == "":
            current_model_dir = current_model_dir / "root"
        else:
            current_model_dir = current_model_dir / hierarchy_root

        save_path = current_model_dir / "gat_model.pt"

        state_dict = torch.load(save_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        graph_ids, logits, preds, labels = predict_with_logits(
            model=model,
            dataloader=loader,
            device=device,
        )

        id2label = {v: k for k, v in dataset.label2id.items()}

        df_logits = pd.DataFrame(
            logits.numpy(),
            columns=[f"logit_{i}" for i in range(logits.shape[1])]
        )

        df_logits.insert(0, "graph_id", graph_ids)

        df_logits["pred_class_id"] = preds.numpy()
        df_logits["y_pred"] = df_logits["pred_class_id"].map(id2label)

        df_logits.drop(columns=["pred_class_id"], inplace=True)

        df_logits.rename(columns={"graph_id": "name"}, inplace=True)

        current_logits_dir = logits_dir
        if hierarchy_root == "":
            current_logits_dir = current_logits_dir / "root"
        else:
            current_logits_dir = current_logits_dir / hierarchy_root

        current_logits_dir.mkdir(parents=True, exist_ok=True)

        out_path = current_logits_dir / "gat.csv"
        df_logits.to_csv(out_path, index=False)

        print(f"Логиты сохранены в {out_path}")