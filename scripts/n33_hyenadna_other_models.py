import io
import json
import pickle
import numpy as np
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from catboost import CatBoostClassifier

import config

name_to_embedding = {}
name_to_type = {}

path_to_hyena_embedding = f'{config.DIR_HYENA}/hyena_embeddings_1024_and_types.pkl'


def load_pickle_torch_cpu(path):
    import torch
    orig = torch.storage._load_from_bytes
    try:
        torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        with open(path, 'rb') as f:
            return pickle.load(f)
    finally:
        torch.storage._load_from_bytes = orig


def root_to_dirname(root: str) -> str:
    return "root" if root == "" else root


data = load_pickle_torch_cpu(path_to_hyena_embedding)
name_to_embedding = data['embeddings']

HIERARCHY_ROOTS = [
    "",
    "Class I (Retrotransposons)",
    "Class II (DNA transposons)",
    "Class I (Retrotransposons)\tLTR Retrotransposon",
    "Class I (Retrotransposons)\tNon-LTR Retrotransposon",
]

with open(f'{config.DIR_REPBASE_PROCESSED}/id_train.txt', 'r', encoding='utf-8') as file:
    names_train = [line.strip() for line in file]

with open(f'{config.DIR_REPBASE_PROCESSED}/id_test.txt', 'r', encoding='utf-8') as file:
    names_test = [line.strip() for line in file]

train_filtered = [name for name in names_train if name in name_to_embedding]
test_filtered = [name for name in names_test if name in name_to_embedding]

with open(
    f'{config.DIR_REPBASE_PROCESSED}/hierarchy_sequences_02_ltr_correction_with_classes.json',
    'r',
    encoding='utf-8'
) as file:
    meta = json.load(file)

for HIERARCHY_ROOT in HIERARCHY_ROOTS:
    print('-' * 20, HIERARCHY_ROOT, '-' * 20)

    save_dir = f"/Users/nad/mobiraph/data/n23_hyena_models/{root_to_dirname(HIERARCHY_ROOT)}"
    model_path = os.path.join(save_dir, "catboost_model.cbm")
    aux_path = os.path.join(save_dir, "catboost_meta.pkl")

    if os.path.exists(model_path) and os.path.exists(aux_path):
        print(f"Модель уже существует: {model_path}. Пропускаю обучение.")
        continue

    current_meta = meta.copy()
    for part in HIERARCHY_ROOT.split("\t"):
        if not part:
            continue
        current_meta = current_meta[part]["subs"]

    name_to_type = {}
    for class_name, class_dict in current_meta.items():
        for seq in class_dict["sequences"]:
            name_to_type[seq] = class_name

    train_filtered_root = [name for name in train_filtered if name in name_to_type]
    test_filtered_root = [name for name in test_filtered if name in name_to_type]

    X_train = np.array([name_to_embedding[name] for name in train_filtered_root], dtype=np.float32)
    y_train = np.array([name_to_type[name] for name in train_filtered_root])

    X_test = np.array([name_to_embedding[name] for name in test_filtered_root], dtype=np.float32)
    y_test = np.array([name_to_type[name] for name in test_filtered_root])

    print('shape X_train', X_train.shape, 'shape y_train', y_train.shape)
    print('shape X_test', X_test.shape, 'shape y_test', y_test.shape)

    if len(X_train) == 0 or len(X_test) == 0:
        print("Пустой train/test для этого уровня. Пропускаю.")
        continue

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    num_classes = len(label_encoder.classes_)
    input_dim = X_train_scaled.shape[1]

    model = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='TotalF1',
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=100,
        auto_class_weights='Balanced'
    )

    model.fit(X_train_scaled, y_train_enc)

    os.makedirs(save_dir, exist_ok=True)
    model.save_model(model_path)

    with open(aux_path, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "label_encoder": label_encoder,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "hierarchy_root": HIERARCHY_ROOT,
        }, f)

    y_pred_enc = model.predict(X_test_scaled).reshape(-1).astype(int)
    y_pred_labels = label_encoder.inverse_transform(y_pred_enc)
    y_true_labels = label_encoder.inverse_transform(y_test_enc)

    raw_logits = model.predict(X_test_scaled, prediction_type="RawFormulaVal")
    raw_logits = np.asarray(raw_logits)

    if raw_logits.ndim == 1:
        raw_logits = raw_logits.reshape(-1, 1)

    class_names = label_encoder.classes_

    df_logits = pd.DataFrame(raw_logits, columns=class_names[:raw_logits.shape[1]])
    df_logits.insert(0, "name", test_filtered_root)
    df_logits["y_true"] = y_true_labels
    df_logits["y_pred"] = y_pred_labels

    logit_dir = f"/Users/nad/mobiraph/data/n22_test_results/{root_to_dirname(HIERARCHY_ROOT)}"
    os.makedirs(logit_dir, exist_ok=True)

    df_logits.to_csv(f"{logit_dir}/hyena_catboost_logits.csv", index=False)

    print(df_logits.head())
    print("Accuracy:", accuracy_score(y_true_labels, y_pred_labels))
    print("F1-macro:", f1_score(y_true_labels, y_pred_labels, average="macro"))