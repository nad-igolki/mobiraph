import json
import os
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

import config


def root_to_dirname(root: str) -> str:
    return "root" if root == "" else root


path_to_onehot_embeddings = "/Users/nad/mobiraph/data/n26_sv_processed/20_symbols_insects.npz"
model_root = "/Users/nad/mobiraph/data/n42_20_symbols_xgb_models"
out_root = "/Users/nad/mobiraph/data/n28_sv_insects_results"

# train_ids_path = "/Users/nad/mobiraph/data/n13_repbase_processed/id_test.txt"
train_ids_path = None
METADATA_PATH = None

HIERARCHY_ROOTS = [
    "",
    "Class I (Retrotransposons)",
    "Class II (DNA transposons)",
    "Class I (Retrotransposons)\tLTR Retrotransposon",
    "Class I (Retrotransposons)\tNon-LTR Retrotransposon",
]


data = np.load(path_to_onehot_embeddings, allow_pickle=True)

names = data["names"]
embeddings = data["embeddings"].reshape(len(data["embeddings"]), -1).astype(np.float32)

name_to_embedding = dict(zip(names, embeddings))


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
    print("-" * 20, hierarchy_root, "-" * 20)

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

    model_path = os.path.join(save_dir, "xgb_model.json")
    classes_path = os.path.join(save_dir, "classes.npy")

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        continue

    if not os.path.exists(classes_path):
        print(f"Classes not found: {classes_path}")
        continue

    model = XGBClassifier()
    model.load_model(model_path)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(classes_path, allow_pickle=True)

    logits = model.predict(X, output_margin=True)

    y_pred_enc = np.argmax(logits, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    df = pd.DataFrame(logits, columns=label_encoder.classes_)
    df.insert(0, "name", selected_names_root)
    df["y_pred"] = y_pred

    if y_raw is not None:
        df["y_true"] = y_raw

    out_dir = os.path.join(out_root, root_to_dirname(hierarchy_root))
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "20_symbols_xgb.csv")
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")