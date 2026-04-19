import io
import json
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config


def load_pickle_torch_cpu(path):
    orig = torch.storage._load_from_bytes
    try:
        torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        with open(path, "rb") as f:
            return pickle.load(f)
    finally:
        torch.storage._load_from_bytes = orig


def root_to_dirname(root: str) -> str:
    return "root" if root == "" else root


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=256, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x[:, 0, :]
        x = self.norm(x)
        return self.classifier(x)


path_to_hyena_embedding = f"{config.DIR_HYENA}/hyena_embeddings_1024_and_types.pkl"
model_root = "/Users/nad/mobiraph/data/n23_hyena_models"
out_root = "/Users/nad/mobiraph/data/n25_train_results"

HIERARCHY_ROOTS = [
    "",
    "Class I (Retrotransposons)",
    "Class II (DNA transposons)",
    "Class I (Retrotransposons)\tLTR Retrotransposon",
    "Class I (Retrotransposons)\tNon-LTR Retrotransposon",
]

data = load_pickle_torch_cpu(path_to_hyena_embedding)
name_to_embedding = data["embeddings"]

with open(f"{config.DIR_REPBASE_PROCESSED}/id_train.txt", "r", encoding="utf-8") as f:
    names_train = [line.strip() for line in f]

with open(f"{config.DIR_REPBASE_PROCESSED}/hierarchy_sequences_02_ltr_correction_with_classes.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

train_filtered = [name for name in names_train if name in name_to_embedding]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for hierarchy_root in HIERARCHY_ROOTS:
    current_meta = meta.copy()
    for part in hierarchy_root.split("\t"):
        if part:
            current_meta = current_meta[part]["subs"]

    name_to_type = {}
    for class_name, class_dict in current_meta.items():
        for seq in class_dict["sequences"]:
            name_to_type[seq] = class_name

    train_filtered_root = [name for name in train_filtered if name in name_to_type]

    X_train = np.array([name_to_embedding[name] for name in train_filtered_root], dtype=np.float32)
    y_train = np.array([name_to_type[name] for name in train_filtered_root])

    save_dir = os.path.join(model_root, root_to_dirname(hierarchy_root))
    checkpoint = torch.load(os.path.join(save_dir, "transformer_model.pt"), map_location=device, weights_only=False)

    scaler = checkpoint["scaler"]
    label_encoder = checkpoint["label_encoder"]
    input_dim = checkpoint["input_dim"]
    num_classes = checkpoint["num_classes"]

    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    y_train_enc = label_encoder.transform(y_train)

    model = TransformerClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=256,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_train_scaled, dtype=torch.float32, device=device)).cpu().numpy()

    y_pred_enc = logits.argmax(axis=1)

    df = pd.DataFrame(logits, columns=label_encoder.classes_)
    df.insert(0, "name", train_filtered_root)
    df["y_true"] = label_encoder.inverse_transform(y_train_enc)
    df["y_pred"] = label_encoder.inverse_transform(y_pred_enc)

    out_dir = os.path.join(out_root, root_to_dirname(hierarchy_root))
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "hyena_transformer_train_logits.csv"), index=False)