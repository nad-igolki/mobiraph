import io
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier

import config


def load_pickle_torch_cpu(path):
    orig = torch.storage._load_from_bytes
    try:
        torch.storage._load_from_bytes = lambda b: torch.load(
            io.BytesIO(b),
            map_location="cpu"
        )
        with open(path, "rb") as f:
            return pickle.load(f)
    finally:
        torch.storage._load_from_bytes = orig


def root_to_dirname(root: str) -> str:
    return "root" if root == "" else root


path_to_hyena_embedding = "/Users/nad/mobiraph/data/n19_hyena_models/hyena_embeddings_1024_and_types.pkl"
model_root = "/Users/nad/mobiraph/data/n13_repbase_processed_wo_bad_sf/hyena"
out_root = "/Users/nad/mobiraph/data/n13_repbase_processed_wo_bad_sf/train_logits"

train_ids_path = '/Users/nad/mobiraph/data/n13_repbase_processed_wo_bad_sf/id_train_with_superfamilies.txt'
METADATA_PATH = None

HIERARCHY_ROOTS = [
    "",
    "Class I (Retrotransposons)",
    "Class II (DNA transposons)",
    "Class I (Retrotransposons)\tLTR Retrotransposon",
    "Class I (Retrotransposons)\tNon-LTR Retrotransposon",
]


data = load_pickle_torch_cpu(path_to_hyena_embedding)
name_to_embedding = data["embeddings"]

if train_ids_path:
    with open(train_ids_path, "r", encoding="utf-8") as f:
        selected_names = [line.strip() for line in f]
else:
    selected_names = list(name_to_embedding.keys())

selected_names = [n for n in selected_names if n in name_to_embedding]

meta = None
if METADATA_PATH:
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)


for hierarchy_root in HIERARCHY_ROOTS:

    if meta is not None:
        current_meta = meta.copy()

        for part in hierarchy_root.split("\t"):
            if part:
                current_meta = current_meta[part]["subs"]

        name_to_type = {
            seq: class_name
            for class_name, class_dict in current_meta.items()
            for seq in class_dict["sequences"]
        }

        selected_names_root = [
            n for n in selected_names
            if n in name_to_type
        ]

        y_raw = np.array([name_to_type[n] for n in selected_names_root])

    else:
        selected_names_root = selected_names
        y_raw = None

    if not selected_names_root:
        print(f"Skip empty root: {hierarchy_root}")
        continue

    X = np.array(
        [name_to_embedding[n] for n in selected_names_root],
        dtype=np.float32
    )

    save_dir = os.path.join(model_root, root_to_dirname(hierarchy_root))

    model_path = os.path.join(save_dir, "catboost_model.cbm")
    meta_path = os.path.join(save_dir, "catboost_meta.pkl")

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        continue

    if not os.path.exists(meta_path):
        print(f"Meta not found: {meta_path}")
        continue

    model = CatBoostClassifier()
    model.load_model(model_path)

    with open(meta_path, "rb") as f:
        catboost_meta = pickle.load(f)

    scaler = catboost_meta["scaler"]
    label_encoder = catboost_meta["label_encoder"]

    X_scaled = scaler.transform(X).astype(np.float32)

    proba = model.predict_proba(X_scaled)

    y_pred_enc = np.argmax(proba, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    df = pd.DataFrame(proba, columns=label_encoder.classes_)
    df.insert(0, "name", selected_names_root)
    df["y_pred"] = y_pred

    if y_raw is not None:
        df["y_true"] = y_raw

    out_dir = os.path.join(out_root, root_to_dirname(hierarchy_root))
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "hyena_catboost.csv")
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")