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
        help="Путь к hierarchy_sequences.json."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.05,
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

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Директория для сохранения всех артефактов."
    )

    parser.add_argument(
        "--hierarchy-root",
        type=str,
        default="",
        help="Корень для классификации."
    )

    parser.add_argument(
        "--train-ids",
        type=str,
        required=True,
        help="Путь к файлу с айдишниками для теста."
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


import numpy as np
import pandas as pd


def prepare_data(
    embeddings_df: pd.DataFrame,
    metadata: dict,
    hierarchy_root: str,
    train_ids: list
):
    if "name" not in embeddings_df.columns:
        raise ValueError("В CSV отсутствует колонка 'name'.")

    emb_cols = [c for c in embeddings_df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("В CSV не найдено колонок, начинающихся с 'emb_'.")

    train_ids_set = set(train_ids)
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

    for class_type, class_info in current_meta.items():
        valid_ids = [
            seq_id for seq_id in class_info["sequences"]
            if seq_id in train_ids_set and seq_id in sequence_ids
        ]
        if not valid_ids:
            continue

        X_list.append(emb_matrix.loc[valid_ids].to_numpy())
        y_list.extend([class_type] * len(valid_ids))

    X = np.vstack(X_list) if X_list else np.array([])
    y = np.array(y_list)

    counts = Counter(y)
    print("Все классы:")
    print(counts)

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


def save_artifacts(
    model,
    label_encoder: LabelEncoder,
    history,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "cnn_model.keras"
    label_encoder_path = output_dir / "label_encoder.pkl"
    history_path = output_dir / "history.json"

    if hasattr(model, "save") and callable(model.save):
        model.save(model_path)
    elif hasattr(model, "model") and hasattr(model.model, "save"):
        model.model.save(model_path)
    else:
        fallback_path = output_dir / "cnn_model.pkl"
        with open(fallback_path, "wb") as f:
            import pickle
            pickle.dump(model, f)
        print(f"Модель сохранена через pickle: {fallback_path}")

    # LabelEncoder
    with open(label_encoder_path, "wb") as f:
        import pickle
        pickle.dump(label_encoder, f)

    # History
    history_data = history.history if hasattr(history, "history") else history
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)

    print(f"Модель сохранена: {model_path}")
    print(f"LabelEncoder сохранён: {label_encoder_path}")
    print(f"History сохранён: {history_path}")


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()

    embeddings_path = Path(args.embeddings_path)
    metadata_path = Path(args.metadata_path)

    embeddings_df = load_embeddings(embeddings_path)
    metadata = load_metadata(metadata_path)

    path_train_ids = args.train_ids
    with open(path_train_ids, "r", encoding="utf-8") as f:
        train_ids = [line.strip() for line in f]

    X, y, le = prepare_data(
        embeddings_df=embeddings_df,
        metadata=metadata,
        train_ids=train_ids,
        hierarchy_root=args.hierarchy_root,
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

    save_artifacts(
        model=model,
        label_encoder=le,
        history=history,
        output_dir=Path(args.output_dir) / args.hierarchy_root if args.hierarchy_root else Path(args.output_dir),
    )

    print("Обучение завершено.")
    print("Классы LabelEncoder:")
    print(list(le.classes_))


if __name__ == "__main__":
    main()