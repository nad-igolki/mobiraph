#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from scripts.n12_cnn_model import CNNClassifierModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Обучение CNN-классификатора на embeddings и metadata."
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
        help="Путь к metadata.json."
    )
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=50,
        help="Минимальное количество объектов в классе, чтобы оставить его."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Доля валидационной выборки."
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed для train/val split."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Количество эпох обучения."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Размер батча."
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


def prepare_data(
    embeddings_df: pd.DataFrame,
    metadata: dict,
    min_class_count: int,
):
    if "name" not in embeddings_df.columns:
        raise ValueError("В CSV отсутствует колонка 'name'.")

    emb_cols = [c for c in embeddings_df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("В CSV не найдено колонок, начинающихся с 'emb_'.")

    sequence_ids = embeddings_df["name"].tolist()

    missing_ids = [seq_id for seq_id in sequence_ids if seq_id not in metadata]
    if missing_ids:
        raise KeyError(
            f"В metadata отсутствуют {len(missing_ids)} id из embeddings. "
            f"Пример: {missing_ids[:5]}"
        )

    missing_type_ids = [
        seq_id for seq_id in sequence_ids
        if "type" not in metadata[seq_id]
    ]
    if missing_type_ids:
        raise KeyError(
            f"У некоторых id в metadata отсутствует поле 'type'. "
            f"Пример: {missing_type_ids[:5]}"
        )

    X = embeddings_df[emb_cols].to_numpy(dtype=np.float32)
    X = X[..., np.newaxis]
    y = [metadata[seq_id]["type"] for seq_id in sequence_ids]

    counts = Counter(y)
    print("Все классы:")
    print(counts)

    rare_classes = {k: v for k, v in counts.items() if v < min_class_count}
    print(f"Классы с количеством < {min_class_count}:")
    print(rare_classes)

    valid_classes = {k for k, v in counts.items() if v >= min_class_count}
    if not valid_classes:
        raise ValueError(
            f"После фильтрации не осталось классов с количеством >= {min_class_count}."
        )

    mask = np.array([label in valid_classes for label in y], dtype=bool)

    X = X[mask]
    y = np.array(y, dtype=object)[mask]

    print("После фильтрации:")
    print("X:", X.shape)
    print("y:", y.shape)
    print("Осталось классов:", len(set(y)))

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
    epochs: int,
    batch_size: int,
):
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("X_train:", X_train.shape, "X_val:", X_val.shape)

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


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()

    embeddings_path = Path(args.embeddings_path)
    metadata_path = Path(args.metadata_path)

    embeddings_df = load_embeddings(embeddings_path)
    metadata = load_metadata(metadata_path)

    X, y, le = prepare_data(
        embeddings_df=embeddings_df,
        metadata=metadata,
        min_class_count=args.min_class_count,
    )

    print("embeddings_df shape:", embeddings_df.shape)
    print("emb columns:", len([c for c in embeddings_df.columns if c.startswith("emb_")]))
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("unique classes:", len(np.unique(y)))
    print("class ids:", np.unique(y)[:20])

    model, history = train_model(
        X=X,
        y=y,
        test_size=args.test_size,
        random_state=args.random_state,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print("Обучение завершено.")
    print("Классы LabelEncoder:")
    print(list(le.classes_))


if __name__ == "__main__":
    main()