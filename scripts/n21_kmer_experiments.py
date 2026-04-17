#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model

from scripts.n12_cnn_model import CNNClassifierModel
from scripts.n10_tables_figures_nice import (
    plot_classification_report,
    plot_confusion_matrix,
)


def prepare_dataset(repo, hierarchy_root: str, sample_ids: list, label_encoder: LabelEncoder = None):
    sample_ids_set = set(sample_ids)
    current_meta = repo.get_meta_node(hierarchy_root)

    X_list = []
    y_list = []

    allowed_classes = set(label_encoder.classes_) if label_encoder is not None else None

    for class_type, class_info in current_meta.items():
        if allowed_classes is not None and class_type not in allowed_classes:
            continue

        valid_ids = [
            seq_id for seq_id in class_info["sequences"]
            if seq_id in sample_ids_set and seq_id in repo.available_ids
        ]

        if not valid_ids:
            continue

        X_list.append(repo.emb_matrix_df.loc[valid_ids].to_numpy())
        y_list.extend([class_type] * len(valid_ids))

    X = np.vstack(X_list)
    y_raw = np.array(y_list)

    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        return X, y, y_raw, label_encoder

    y = label_encoder.transform(y_raw)
    return X, y, y_raw


def train_model(X, y, epochs: int, batch_size: int, test_size: float, random_state: int):
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    input_dim = X_train.shape[1]
    class_num = len(np.unique(y))

    model = CNNClassifierModel(input_dim=input_dim, class_num=class_num)
    history = model.train(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
    )

    return model, history


def save_model_artifacts(model, label_encoder, history, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "cnn_model.keras"
    label_encoder_path = output_dir / "label_encoder.pkl"
    history_path = output_dir / "history.json"

    if hasattr(model, "save"):
        model.save(model_path)
    else:
        model.model.save(model_path)

    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    history_data = history.history if hasattr(history, "history") else history
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)


def predict_with_model(model, X: np.ndarray, batch_size: int = 32):
    if hasattr(model, "model") and hasattr(model.model, "predict"):
        probs = model.model.predict(X, batch_size=batch_size)
    elif hasattr(model, "predict"):
        probs = model.predict(X)
    else:
        raise AttributeError("У модели нет метода predict().")

    if isinstance(probs, (list, tuple)):
        probs = probs[0]

    probs = np.asarray(probs)

    if probs.ndim == 1:
        y_pred = (probs > 0.5).astype(int)
    else:
        y_pred = np.argmax(probs, axis=1)

    return probs, y_pred


def evaluate_with_model(model, X: np.ndarray, y: np.ndarray, batch_size: int = 32):
    loss, acc = None, None

    try:
        if hasattr(model, "evaluate"):
            result = model.evaluate(X, y, batch_size=batch_size)
        elif hasattr(model, "model") and hasattr(model.model, "evaluate"):
            result = model.model.evaluate(X, y, batch_size=batch_size)
        else:
            return loss, acc

        if isinstance(result, (list, tuple)):
            if len(result) >= 2:
                loss, acc = result[0], result[1]
            elif len(result) == 1:
                loss = result[0]
        else:
            loss = result
    except Exception as e:
        print(f"Не удалось выполнить evaluate(): {e}")

    return loss, acc


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    output_dir: Path,
    batch_size: int = 32,
):
    loss, acc = evaluate_with_model(
        model=model,
        X=X_test,
        y=y_test,
        batch_size=batch_size,
    )

    probs, y_pred = predict_with_model(
        model=model,
        X=X_test,
        batch_size=batch_size,
    )

    y_true_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    report_text = classification_report(y_true_labels, y_pred_labels, digits=4)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    plot_classification_report(
        y_true_labels,
        y_pred_labels,
        save_path=output_dir / "classification_report.png"
    )

    plot_confusion_matrix(
        y_true_labels,
        y_pred_labels,
        labels=label_encoder.classes_,
        save_path=output_dir / "confusion_matrix.png",
        normalize=False
    )

    return loss, acc, report_text


def root_to_dirname(root: str) -> str:
    if root == "":
        return "root"
    return root.replace(" ", "_").replace("/", "_")


def root_to_dirname(root: str) -> str:
    if root == "":
        return "root"
    return root.replace(" ", "_").replace("/", "_")