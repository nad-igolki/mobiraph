import os
import json
import numpy as np
import pandas as pd
import config

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score


def root_to_dirname(root: str) -> str:
    return "root" if root == "" else root


# --- one-hot embeddings ---
data = np.load("/Users/nad/mobiraph/data/n13_repbase_processed/20_symbols.npz", allow_pickle=True)

names = data["names"]
embeddings = data["embeddings"].reshape(len(data["embeddings"]), -1).astype(np.float32)

name_to_embedding = dict(zip(names, embeddings))


# --- hierarchy roots ---
HIERARCHY_ROOTS = [
    "",
    "Class I (Retrotransposons)",
    "Class II (DNA transposons)",
    "Class I (Retrotransposons)\tLTR Retrotransposon",
    "Class I (Retrotransposons)\tNon-LTR Retrotransposon",
]


# --- train/test ids ---
with open(f"{config.DIR_REPBASE_PROCESSED}/id_train.txt", "r", encoding="utf-8") as f:
    names_train = [line.strip() for line in f]

with open(f"{config.DIR_REPBASE_PROCESSED}/id_test.txt", "r", encoding="utf-8") as f:
    names_test = [line.strip() for line in f]


train_filtered = [name for name in names_train if name in name_to_embedding]
test_filtered = [name for name in names_test if name in name_to_embedding]


# --- meta ---
with open(
    f"{config.DIR_REPBASE_PROCESSED}/hierarchy_sequences_02_ltr_correction_with_classes.json",
    "r",
    encoding="utf-8"
) as f:
    meta = json.load(f)


params = {
    "subsample": 1.0,
    "reg_lambda": 2,
    "reg_alpha": 0.01,
    "n_estimators": 800,
    "max_depth": 8,
    "learning_rate": 0.03,
    "gamma": 0.3,
    "colsample_bytree": 0.6,
}


for HIERARCHY_ROOT in HIERARCHY_ROOTS:
    print("-" * 20, HIERARCHY_ROOT, "-" * 20)

    save_dir = f"/Users/nad/mobiraph/data/n42_20_symbols_xgb_models/{root_to_dirname(HIERARCHY_ROOT)}"
    model_path = os.path.join(save_dir, "xgb_model.json")

    if os.path.exists(model_path):
        print(f"Модель уже существует: {model_path}. Пропускаю обучение.")
        continue

    current_meta = meta.copy()

    for part in HIERARCHY_ROOT.split("\t"):
        if part:
            current_meta = current_meta[part]["subs"]

    name_to_type = {}

    for class_name, class_dict in current_meta.items():
        for seq in class_dict["sequences"]:
            name_to_type[seq] = class_name

    train_filtered_root = [
        name for name in train_filtered
        if name in name_to_type
    ]

    test_filtered_root = [
        name for name in test_filtered
        if name in name_to_type
    ]

    X_train = np.array(
        [name_to_embedding[name] for name in train_filtered_root],
        dtype=np.float32
    )
    y_train = np.array([name_to_type[name] for name in train_filtered_root])

    X_test = np.array(
        [name_to_embedding[name] for name in test_filtered_root],
        dtype=np.float32
    )
    y_test = np.array([name_to_type[name] for name in test_filtered_root])

    print("shape X_train", X_train.shape, "shape y_train", y_train.shape)
    print("shape X_test", X_test.shape, "shape y_test", y_test.shape)

    label_encoder = LabelEncoder()

    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    num_classes = len(np.unique(y_train_enc))

    model = XGBClassifier(
        **params,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train_enc)

    y_pred_enc = np.argmax(model.predict_proba(X_test), axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    logits = model.predict(X_test, output_margin=True)  # (N, num_classes)
    class_names = label_encoder.classes_

    df_proba = pd.DataFrame(logits, columns=class_names)
    df_proba.insert(0, "name", test_filtered_root)
    df_proba["y_true"] = y_test
    df_proba["y_pred"] = y_pred

    os.makedirs(save_dir, exist_ok=True)

    model.save_model(model_path)

    np.save(os.path.join(save_dir, "classes.npy"), label_encoder.classes_)

    result_dir = f"/Users/nad/mobiraph/data/n22_test_results/{root_to_dirname(HIERARCHY_ROOT)}"
    os.makedirs(result_dir, exist_ok=True)

    df_proba.to_csv(f"{result_dir}/onehot_xgb_logits.csv", index=False)

    print(df_proba.head())

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1-macro:", f1_score(y_test, y_pred, average="macro"))
    print("F1-weighted:", f1_score(y_test, y_pred, average="weighted"))