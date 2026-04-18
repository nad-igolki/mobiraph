import pandas as pd
import numpy as np
from functools import reduce
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib


def prepare_base_file(df: pd.DataFrame) -> pd.DataFrame:
    required = {"name", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В первом файле не хватает колонок: {missing}")

    df = df.copy()
    feature_cols = [c for c in df.columns if c not in ["name", "y_true", "y_pred"]]
    renamed = {c: f"base__{c}" for c in feature_cols}
    df = df.rename(columns=renamed)

    return df


def prepare_extra_file(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    if "name" not in df.columns:
        raise ValueError(f"В дополнительном файле #{idx} нет колонки 'name'")

    df = df.copy()
    feature_cols = [c for c in df.columns if c != "name"]
    renamed = {c: f"extra{idx}__{c}" for c in feature_cols}
    df = df.rename(columns=renamed)

    return df


def build_merged_dataset(base_df: pd.DataFrame, extra_dfs: list[pd.DataFrame]):
    base_df = prepare_base_file(base_df)

    prepared = [base_df]
    for i, df in enumerate(extra_dfs, start=1):
        prepared.append(prepare_extra_file(df, i))

    merged = reduce(lambda l, r: l.merge(r, on="name", how="left"), prepared)

    classes = sorted(
        set(merged["y_true"].dropna().astype(str).unique()).union(
            set(merged["y_pred"].dropna().astype(str).unique())
        )
    )

    return merged, classes


def train_on_concatenated_embeddings(
    base_df: pd.DataFrame,
    extra_dfs: list[pd.DataFrame],
    target_col: str = "y_true",
    test_size: float = 0.2,
    random_state: int = 42,
    save_path: str = "/Users/nad/mobiraph/data/n24_ensemble_models/model_bundle_class2.pkl",   # <-- путь для сохранения
):
    merged, classes = build_merged_dataset(base_df, extra_dfs)

    merged = merged[merged[target_col].notna()].copy()

    feature_cols = [c for c in merged.columns if c not in ["name", "y_true", "y_pred"]]

    X = merged[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = merged[target_col].astype(str)

    le = LabelEncoder()
    le.fit(classes)

    unknown_targets = sorted(set(y.unique()) - set(classes))
    if unknown_targets:
        raise ValueError(f"Лишние классы в y_true: {unknown_targets}")

    y_enc = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred_enc = model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)
    y_test_labels = le.inverse_transform(y_test)

    print(classification_report(y_test_labels, y_pred, labels=list(le.classes_), zero_division=0))

    bundle = {
        "model": model,
        "label_encoder": le,
        "classes": list(le.classes_),
        "feature_cols": feature_cols,
    }

    # ---- СОХРАНЕНИЕ ----
    joblib.dump(bundle, save_path)
    print(f"Модель сохранена в: {save_path}")

    return bundle


def predict_on_new_data(bundle, base_df: pd.DataFrame, extra_dfs: list[pd.DataFrame]):
    merged, _ = build_merged_dataset(base_df, extra_dfs)

    feature_cols = bundle["feature_cols"]
    le = bundle["label_encoder"]
    model = bundle["model"]

    for col in feature_cols:
        if col not in merged.columns:
            merged[col] = 0.0

    X = merged[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    pred = le.inverse_transform(model.predict(X))

    out = merged[["name"]].copy()
    out["prediction"] = pred

    return out

extra5 = pd.read_csv("/Users/nad/mobiraph/data/n22_test_results/root/hyena_transformer.csv")      # тут есть name, y_true, y_pred
extra1 = pd.read_csv("/Users/nad/mobiraph/data/n22_test_results/root/kmer_cnn.csv")       # тут есть name + любые другие колонки
extra2 = pd.read_csv("/Users/nad/mobiraph/data/n22_test_results/Class II (DNA transposons)/hyena_transformer.csv")
base_df = pd.read_csv("/Users/nad/mobiraph/data/n22_test_results/Class II (DNA transposons)/kmer_cnn.csv")
extra4 = pd.read_csv("/Users/nad/mobiraph/data/n22_test_results/Class I (Retrotransposons)/hyena_transformer.csv")
extra3 = pd.read_csv("/Users/nad/mobiraph/data/n22_test_results/Class I (Retrotransposons)/kmer_cnn.csv")

bundle = train_on_concatenated_embeddings(
    base_df=base_df,
    extra_dfs=[extra1, extra2],
    target_col="y_true"
)

preds = predict_on_new_data(
    bundle,
    base_df=base_df,
    extra_dfs=[extra1, extra2]
)

print(preds.head())