import pandas as pd
import joblib
from functools import reduce


def prepare_base_file(df: pd.DataFrame) -> pd.DataFrame:
    if "name" not in df.columns:
        raise ValueError("В первом файле нет колонки 'name'")

    df = df.copy()
    feature_cols = [c for c in df.columns if c not in ["name", "y_true", "y_pred"]]
    renamed = {c: f"base__{c}" for c in feature_cols}
    return df.rename(columns=renamed)


def prepare_extra_file(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    if "name" not in df.columns:
        raise ValueError(f"В дополнительном файле #{idx} нет колонки 'name'")

    df = df.copy()
    feature_cols = [c for c in df.columns if c != "name"]
    renamed = {c: f"extra{idx}__{c}" for c in feature_cols}
    return df.rename(columns=renamed)


def build_merged_dataset(base_df: pd.DataFrame, extra_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    prepared = [prepare_base_file(base_df)]
    for i, df in enumerate(extra_dfs, start=1):
        prepared.append(prepare_extra_file(df, i))

    return reduce(lambda l, r: l.merge(r, on="name", how="left"), prepared)


def predict_on_new_data(bundle, base_df: pd.DataFrame, extra_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    merged = build_merged_dataset(base_df, extra_dfs)

    feature_cols = bundle["feature_cols"]
    le = bundle["label_encoder"]
    model = bundle["model"]

    for col in feature_cols:
        if col not in merged.columns:
            merged[col] = 0.0

    X = merged[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_pred = le.inverse_transform(model.predict(X))

    out = merged[["name"]].copy()
    out["y_pred"] = y_pred
    return out


results_dir = "/Users/nad/mobiraph/data/n29_sv_insects_results_30"
ensemble_dir = "/Users/nad/mobiraph/data/n24_ensemble_models"

HIERARCHY_ROOTS = [
    "root",
    "Class I (Retrotransposons)",
    "Class II (DNA transposons)",
    "Class I (Retrotransposons)\tLTR Retrotransposon",
    "Class I (Retrotransposons)\tNon-LTR Retrotransposon",
]

for i, hierarchy_root in enumerate(HIERARCHY_ROOTS):
    base_df = pd.read_csv(f"{results_dir}/{hierarchy_root}/hyena_transformer.csv")
    extra_dfs = [pd.read_csv(f"{results_dir}/{hierarchy_root}/kmer_cnn.csv")]

    for j in range(1, 5):
        idx = (i + j) % 5
        extra_dfs.append(pd.read_csv(f"{results_dir}/{HIERARCHY_ROOTS[idx]}/hyena_transformer.csv"))
        extra_dfs.append(pd.read_csv(f"{results_dir}/{HIERARCHY_ROOTS[idx]}/kmer_cnn.csv"))

    model_path = f"{ensemble_dir}/train_{hierarchy_root}.pkl"
    bundle = joblib.load(model_path)

    preds = predict_on_new_data(
        bundle=bundle,
        base_df=base_df,
        extra_dfs=extra_dfs,
    )

    preds.to_csv(f"{results_dir}/{hierarchy_root}/ensemble.csv", index=False)