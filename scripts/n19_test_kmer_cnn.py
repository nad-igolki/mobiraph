#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import pickle
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import load_model

from scripts.n10_tables_figures_nice import plot_classification_report, plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Тест CNN-классификатора на embeddings и metadata."
    )
    parser.add_argument(
        "--embeddings-path",
        type=str,
        required=True,
        help="Путь к CSV с embeddings."
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        required=True,
        help="Путь к hierarchy_sequences.json."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Путь к сохранённой модели (.keras или .pkl)."
    )
    parser.add_argument(
        "--label-encoder-path",
        type=str,
        required=True,
        help="Путь к label_encoder.pkl."
    )
    parser.add_argument(
        "--test-ids",
        type=str,
        required=True,
        help="Путь к файлу с id для теста."
    )
    parser.add_argument(
        "--hierarchy-root",
        type=str,
        default="",
        help="Корень для классификации."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Размер батча для predict/evaluate."
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        required=False,
        help="Директория для сохранения графиков."
    )
    return parser.parse_args()


def load_embeddings(embeddings_path: Path) -> pd.DataFrame:
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Файл embeddings не найден: {embeddings_path}")
    return pd.read_csv(embeddings_path)


def load_metadata(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Файл metadata не найден: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_label_encoder(label_encoder_path: Path) -> LabelEncoder:
    if not label_encoder_path.exists():
        raise FileNotFoundError(f"Файл LabelEncoder не найден: {label_encoder_path}")
    with open(label_encoder_path, "rb") as f:
        return pickle.load(f)


def load_test_ids(test_ids_path: Path) -> list:
    if not test_ids_path.exists():
        raise FileNotFoundError(f"Файл с test ids не найден: {test_ids_path}")
    with open(test_ids_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def prepare_data(
    embeddings_df: pd.DataFrame,
    metadata: dict,
    hierarchy_root: str,
    sample_ids: list,
    label_encoder: LabelEncoder,
):
    if "name" not in embeddings_df.columns:
        raise ValueError("В CSV отсутствует колонка 'name'.")

    emb_cols = [c for c in embeddings_df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("В CSV не найдено колонок, начинающихся с 'emb_'.")

    sample_ids_set = set(sample_ids)
    sequence_ids = embeddings_df["name"].tolist()

    indexed_df = embeddings_df.set_index("name")
    emb_matrix = indexed_df[emb_cols].astype(np.float32)
    emb_matrix = emb_matrix[~emb_matrix.index.duplicated(keep="first")]

    X_list = []
    y_list = []

    current_meta = metadata
    for path_part in hierarchy_root.split("\t"):
        if not path_part:
            break
        current_meta = current_meta[path_part]["subs"]

    allowed_classes = set(label_encoder.classes_)

    for class_type, class_info in current_meta.items():
        if class_type not in allowed_classes:
            continue

        valid_ids = [
            seq_id for seq_id in class_info["sequences"]
            if seq_id in sample_ids_set and seq_id in sequence_ids
        ]
        if not valid_ids:
            continue

        X_list.append(emb_matrix.loc[valid_ids].to_numpy())
        y_list.extend([class_type] * len(valid_ids))

    if not X_list:
        raise ValueError("После фильтрации не найдено ни одного тестового объекта.")

    X = np.vstack(X_list)
    y = np.array(y_list)

    print("Все классы в тесте:")
    print(Counter(y))

    y_encoded = label_encoder.transform(y)

    print("После фильтрации:")
    print("X:", X.shape)
    print("y:", y.shape)
    print("Осталось классов:", len(set(y)))

    return X, y_encoded, y


def load_trained_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")

    if model_path.suffix == ".keras":
        return load_model(model_path)

    if model_path.suffix == ".pkl":
        with open(model_path, "rb") as f:
            return pickle.load(f)

    raise ValueError(f"Неподдерживаемый формат модели: {model_path.suffix}")


def predict_with_model(model, X: np.ndarray, batch_size: int = 32):
    if hasattr(model, "predict"):
        probs = model.predict(X, batch_size=batch_size, verbose=0)
    elif hasattr(model, "model") and hasattr(model.model, "predict"):
        probs = model.model.predict(X, batch_size=batch_size, verbose=0)
    else:
        raise AttributeError("У модели нет метода predict().")

    if probs.ndim == 1:
        y_pred = (probs > 0.5).astype(int)
    else:
        y_pred = np.argmax(probs, axis=1)

    return probs, y_pred


def evaluate_model(model, X: np.ndarray, y: np.ndarray, batch_size: int = 32):
    loss, acc = None, None

    try:
        if hasattr(model, "evaluate"):
            result = model.evaluate(X, y, batch_size=batch_size, verbose=0)
        elif hasattr(model, "model") and hasattr(model.model, "evaluate"):
            result = model.model.evaluate(X, y, batch_size=batch_size, verbose=0)
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


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()

    embeddings_df = load_embeddings(Path(args.embeddings_path))
    metadata = load_metadata(Path(args.metadata_path))
    label_encoder = load_label_encoder(Path(args.label_encoder_path))
    test_ids = load_test_ids(Path(args.test_ids))
    model = load_trained_model(Path(args.model_path))
    plots_dir = Path(args.plots_dir) if args.plots_dir else None

    X_test, y_test, y_test_raw = prepare_data(
        embeddings_df=embeddings_df,
        metadata=metadata,
        hierarchy_root=args.hierarchy_root,
        sample_ids=test_ids,
        label_encoder=label_encoder,
    )

    print("embeddings_df shape:", embeddings_df.shape)
    print("emb columns:", len([c for c in embeddings_df.columns if c.startswith("emb_")]))
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("unique classes:", len(np.unique(y_test)))
    print("class ids:", np.unique(y_test)[:20])

    loss, acc = evaluate_model(
        model=model,
        X=X_test,
        y=y_test,
        batch_size=args.batch_size,
    )

    probs, y_pred = predict_with_model(
        model=model,
        X=X_test,
        batch_size=args.batch_size,
    )

    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_true_labels = label_encoder.inverse_transform(y_test)

    if loss is not None:
        print(f"Test loss: {loss:.6f}")
    if acc is not None:
        print(f"Test accuracy: {acc:.6f}")

    print("\nClassification report:")
    print(classification_report(y_true_labels, y_pred_labels, digits=4))
    if plots_dir:
        plot_classification_report(
            y_true_labels,
            y_pred_labels,
            save_path=plots_dir / "classification_report_test.png"
        )
        plot_confusion_matrix(
            y_true_labels,
            y_pred_labels,
            labels=label_encoder.classes_,
            save_path=plots_dir / "confusion_matrix_test.png",
            normalize=False
        )
    else:
        plot_classification_report(
            y_true_labels,
            y_pred_labels
        )

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true_labels, y_pred_labels, labels=label_encoder.classes_))


    print("\nКлассы LabelEncoder:")
    print(list(label_encoder.classes_))


if __name__ == "__main__":
    main()