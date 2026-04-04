#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import pickle
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from scripts.n12_cnn_model import CNNClassifierModel


def safe_dir_name(name: str) -> str:
    return name.replace("/", "／").replace("\\", "＼")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Обучение CNN-моделей на каждом узле иерархии."
    )

    parser.add_argument(
        "--embeddings-path",
        type=str,
        required=True,
        help="Путь к CSV с embeddings."
    )
    parser.add_argument(
        "--hierarchy-json-path",
        type=str,
        required=True,
        help="Путь к JSON с иерархией, где у каждого узла есть sequences и subs."
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
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts_hierarchy",
        help="Корневая директория для сохранения моделей по иерархии."
    )
    parser.add_argument(
        "--skip-self-class",
        action="store_true",
        help=(
            "Если указан, последовательности узла, не принадлежащие ни одному "
            "непосредственному ребёнку, будут пропущены, а не помечены классом __self__."
        )
    )

    return parser.parse_args()


def load_embeddings(embeddings_path: Path) -> pd.DataFrame:
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Файл embeddings не найден: {embeddings_path}")
    return pd.read_csv(embeddings_path)


def load_hierarchy(hierarchy_path: Path) -> dict:
    if not hierarchy_path.exists():
        raise FileNotFoundError(f"Файл hierarchy json не найден: {hierarchy_path}")
    with open(hierarchy_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_embeddings_df(embeddings_df: pd.DataFrame) -> List[str]:
    if "name" not in embeddings_df.columns:
        raise ValueError("В CSV отсутствует колонка 'name'.")

    emb_cols = [c for c in embeddings_df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("В CSV не найдено колонок, начинающихся с 'emb_'.")

    return emb_cols


def build_virtual_root_node(hierarchy: Dict) -> Dict:
    all_sequences = []

    for node in hierarchy.values():
        all_sequences.extend(node.get("sequences", []))

    seen = set()
    unique_sequences = []
    for seq_id in all_sequences:
        if seq_id not in seen:
            seen.add(seq_id)
            unique_sequences.append(seq_id)

    return {
        "sequences": unique_sequences,
        "subs": hierarchy,
    }


def build_embedding_index(
    embeddings_df: pd.DataFrame,
    emb_cols: List[str]
) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Возвращает:
    - id -> индекс строки в embeddings_df
    - массив embeddings формы (N, D, 1)
    """
    ids = embeddings_df["name"].astype(str).tolist()

    # duplicated = embeddings_df["name"][embeddings_df["name"].duplicated()].tolist()
    # if duplicated:
    #     raise ValueError(
    #         f"В embeddings найдены дублирующиеся id. Пример: {len(duplicated)}"
    #     )

    id_to_index = {seq_id: i for i, seq_id in enumerate(ids)}
    X_all = embeddings_df[emb_cols].to_numpy(dtype=np.float32)
    X_all = X_all[..., np.newaxis]

    return id_to_index, X_all


def get_node_by_path(hierarchy: Dict, path: Tuple[str, ...]) -> Dict:
    current_level = hierarchy
    node = None

    for name in path:
        if name not in current_level:
            raise KeyError(f"Путь {'/'.join(path)} не найден в hierarchy json.")
        node = current_level[name]
        current_level = node.get("subs", {})

    if node is None:
        raise ValueError("Получен пустой узел.")
    return node


def iter_nodes(hierarchy: Dict):
    """
    Итерирует по всем узлам дерева.
    Возвращает:
    - path_tuple
    - node_dict
    """
    def walk(subtree: Dict, ancestors: List[str]):
        for class_name, node in subtree.items():
            current_path = ancestors + [class_name]
            yield tuple(current_path), node

            subs = node.get("subs", {})
            if subs:
                yield from walk(subs, current_path)

    yield from walk(hierarchy, [])


def prepare_node_dataset(
    node_path: Tuple[str, ...],
    node: Dict,
    id_to_index: Dict[str, int],
    X_all: np.ndarray,
    min_class_count: int,
    skip_self_class: bool = False,
):
    """
    Для узла строит локальный датасет:
    X = embeddings для последовательностей данного узла
    y = имя непосредственного ребёнка, в чьём поддереве лежит id
        или __self__, если id принадлежит текущему узлу, но не попал ни в одного ребёнка

    Возвращает:
    - X
    - y_encoded
    - label_encoder
    - info dict
    или None, если модель для узла обучать не нужно.
    """
    subs = node.get("subs", {})
    if not subs:
        return None

    node_seq_ids = node.get("sequences", [])
    if not node_seq_ids:
        return None

    child_to_ids = {
        child_name: set(child_node.get("sequences", []))
        for child_name, child_node in subs.items()
    }

    X_indices = []
    y_labels = []

    missing_in_embeddings = []
    ambiguous_ids = []

    for seq_id in node_seq_ids:
        matched_children = [
            child_name
            for child_name, child_ids in child_to_ids.items()
            if seq_id in child_ids
        ]

        if len(matched_children) > 1:
            ambiguous_ids.append(seq_id)
            continue

        if len(matched_children) == 1:
            label = matched_children[0]
        else:
            continue

        if seq_id not in id_to_index:
            missing_in_embeddings.append(seq_id)
            continue

        X_indices.append(id_to_index[seq_id])
        y_labels.append(label)

    if ambiguous_ids:
        print(
            f"[WARN] Узел {'/'.join(node_path)}: "
            f"{len(ambiguous_ids)} id попали сразу в несколько непосредственных детей. "
            f"Они будут пропущены. Пример: {ambiguous_ids[:5]}"
        )

    if not X_indices:
        print(f"[SKIP] Узел {'/'.join(node_path)}: нет данных после сопоставления ids.")
        return None

    counts_before = Counter(y_labels)
    print(f"\n=== Узел {'/'.join(node_path)} ===")
    print("Классы до фильтрации:")
    print(counts_before)

    valid_classes = {k for k, v in counts_before.items() if v >= min_class_count}
    rare_classes = {k: v for k, v in counts_before.items() if v < min_class_count}

    print(f"Классы с количеством < {min_class_count}:")
    print(rare_classes)

    if len(valid_classes) < 2:
        print(
            f"[SKIP] Узел {'/'.join(node_path)}: "
            f"после фильтрации осталось меньше 2 классов."
        )
        return None

    filtered_indices = []
    filtered_labels = []

    for idx, label in zip(X_indices, y_labels):
        if label in valid_classes:
            filtered_indices.append(idx)
            filtered_labels.append(label)

    X = X_all[np.array(filtered_indices, dtype=int)]
    y = np.array(filtered_labels, dtype=object)

    if len(set(y)) < 2:
        print(
            f"[SKIP] Узел {'/'.join(node_path)}: "
            f"после финальной фильтрации меньше 2 классов."
        )
        return None

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    info = {
        "node_path": list(node_path),
        "node_name": node_path[-1],
        "total_node_sequences": len(node_seq_ids),
        "used_sequences": len(filtered_indices),
        "missing_in_embeddings_count": len(missing_in_embeddings),
        "missing_in_embeddings_example": missing_in_embeddings[:10],
        "class_counts_before_filter": dict(counts_before),
        "valid_classes_after_filter": sorted(valid_classes),
        "class_counts_after_filter": dict(Counter(y)),
    }

    print("После фильтрации:")
    print("X:", X.shape)
    print("y:", y.shape)
    print("Осталось классов:", len(set(y)))
    print("Классы:", list(le.classes_))

    return X, y_encoded, le, info


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
    epochs: int,
    batch_size: int,
):
    if len(np.unique(y)) < 2:
        raise ValueError("Для обучения нужно как минимум 2 класса.")

    class_counts = Counter(y.tolist())
    min_count = min(class_counts.values())

    if min_count < 2:
        raise ValueError(
            "Хотя бы один класс содержит меньше 2 объектов, "
            "stratify train_test_split невозможен."
        )

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
    extra_info: Optional[dict] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "cnn_model.keras"
    label_encoder_path = output_dir / "label_encoder.pkl"
    history_path = output_dir / "history.json"
    info_path = output_dir / "node_info.json"

    if hasattr(model, "save") and callable(model.save):
        model.save(model_path)
    elif hasattr(model, "model") and hasattr(model.model, "save"):
        model.model.save(model_path)
    else:
        fallback_path = output_dir / "cnn_model.pkl"
        with open(fallback_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Модель сохранена через pickle: {fallback_path}")

    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    history_data = history.history if hasattr(history, "history") else history
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)

    if extra_info is not None:
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(extra_info, f, ensure_ascii=False, indent=2)

    print(f"Модель сохранена: {model_path}")
    print(f"LabelEncoder сохранён: {label_encoder_path}")
    print(f"History сохранён: {history_path}")
    if extra_info is not None:
        print(f"Node info сохранён: {info_path}")


def train_hierarchy_models(
    embeddings_df: pd.DataFrame,
    hierarchy: Dict,
    output_dir: Path,
    min_class_count: int,
    test_size: float,
    random_state: int,
    epochs: int,
    batch_size: int,
    skip_self_class: bool,
):
    emb_cols = validate_embeddings_df(embeddings_df)
    id_to_index, X_all = build_embedding_index(embeddings_df, emb_cols)

    summary = {
        "trained_nodes": [],
        "skipped_nodes": [],
    }

    print("\n=== Обучение ROOT-модели ===")

    virtual_root = build_virtual_root_node(hierarchy)

    try:
        prepared = prepare_node_dataset(
            node_path=("ROOT",),
            node=virtual_root,
            id_to_index=id_to_index,
            X_all=X_all,
            min_class_count=min_class_count,
        )

        if prepared is not None:
            X, y, le, info = prepared

            model, history = train_model(
                X=X,
                y=y,
                test_size=test_size,
                random_state=random_state,
                epochs=epochs,
                batch_size=batch_size,
            )

            root_output_dir = output_dir / "_root_"

            save_artifacts(
                model=model,
                label_encoder=le,
                history=history,
                output_dir=root_output_dir,
                extra_info=info,
            )

            summary["trained_nodes"].append({
                "node_path": ["ROOT"],
                "output_dir": str(root_output_dir),
                "classes": list(le.classes_),
                "used_sequences": info["used_sequences"],
            })

        else:
            summary["skipped_nodes"].append({
                "node_path": ["ROOT"],
                "reason": "dataset_not_suitable"
            })

    except Exception as e:
        print(f"[ERROR] Узел ROOT: {e}")
        summary["skipped_nodes"].append({
            "node_path": ["ROOT"],
            "reason": f"error: {str(e)}"
        })

    for node_path, node in iter_nodes(hierarchy):
        try:
            prepared = prepare_node_dataset(
                node_path=node_path,
                node=node,
                id_to_index=id_to_index,
                X_all=X_all,
                min_class_count=min_class_count,
                skip_self_class=skip_self_class,
            )

            if prepared is None:
                summary["skipped_nodes"].append({
                    "node_path": list(node_path),
                    "reason": "dataset_not_suitable"
                })
                continue

            X, y, le, info = prepared

            model, history = train_model(
                X=X,
                y=y,
                test_size=test_size,
                random_state=random_state,
                epochs=epochs,
                batch_size=batch_size,
            )

            node_output_dir = output_dir.joinpath(
                *[safe_dir_name(x) for x in node_path]
            )

            save_artifacts(
                model=model,
                label_encoder=le,
                history=history,
                output_dir=node_output_dir,
                extra_info=info,
            )

            summary["trained_nodes"].append({
                "node_path": list(node_path),
                "output_dir": str(node_output_dir),
                "classes": list(le.classes_),
                "used_sequences": info["used_sequences"],
            })

        except Exception as e:
            print(f"[ERROR] Узел {'/'.join(node_path)}: {e}")
            summary["skipped_nodes"].append({
                "node_path": list(node_path),
                "reason": f"error: {str(e)}"
            })

    summary_path = output_dir / "training_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== ИТОГО ===")
    print(f"Обучено узлов: {len(summary['trained_nodes'])}")
    print(f"Пропущено узлов: {len(summary['skipped_nodes'])}")
    print(f"Сводка сохранена: {summary_path}")


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()

    embeddings_path = Path(args.embeddings_path)
    hierarchy_json_path = Path(args.hierarchy_json_path)
    output_dir = Path(args.output_dir)

    embeddings_df = load_embeddings(embeddings_path)
    hierarchy = load_hierarchy(hierarchy_json_path)

    print("embeddings_df shape:", embeddings_df.shape)
    print("emb columns:", len([c for c in embeddings_df.columns if c.startswith("emb_")]))

    train_hierarchy_models(
        embeddings_df=embeddings_df,
        hierarchy=hierarchy,
        output_dir=output_dir,
        min_class_count=args.min_class_count,
        test_size=args.test_size,
        random_state=args.random_state,
        epochs=args.epochs,
        batch_size=args.batch_size,
        skip_self_class=args.skip_self_class,
    )

    print("Обучение иерархии завершено.")


if __name__ == "__main__":
    main()