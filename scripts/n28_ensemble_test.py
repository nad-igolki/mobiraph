import pandas as pd
import joblib
from sklearn.metrics import classification_report
from functools import reduce


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


results_dir = "/Users/nad/mobiraph/data/n28_sv_insects_results"
HIERARCHY_ROOTS = [
        "root",
        "Class I (Retrotransposons)",
        "Class II (DNA transposons)",
        "Class I (Retrotransposons)\tLTR Retrotransposon",
        "Class I (Retrotransposons)\tNon-LTR Retrotransposon",
    ]

for i in range(len(HIERARCHY_ROOTS)):
    hierarchy_root = HIERARCHY_ROOTS[i]

    base_df = pd.read_csv(f"{results_dir}/{hierarchy_root}/hyena_transformer.csv")
    extra1 = pd.read_csv(f"{results_dir}/{hierarchy_root}/kmer_cnn.csv")
    extra_dfs = [extra1]
    for j in range(1, 5):
        ind_in_hierarchy = (i + j) % 5
        extra_dfs.append(pd.read_csv(f"{results_dir}/{HIERARCHY_ROOTS[ind_in_hierarchy]}/hyena_transformer.csv"))
        extra_dfs.append(pd.read_csv(f"{results_dir}/{HIERARCHY_ROOTS[ind_in_hierarchy]}/kmer_cnn.csv"))


    model_path = f"/Users/nad/mobiraph/data/n24_ensemble_models/train_{hierarchy_root}.pkl"
    bundle = joblib.load(model_path)


    preds = predict_on_new_data(
        bundle=bundle,
        base_df=base_df,
        extra_dfs=extra_dfs,
    )

    res = base_df[["name", "y_true"]].merge(preds, on="name", how="left")

    print(classification_report(
        res["y_true"].astype(str),
        res["prediction"].astype(str),
        labels=bundle["classes"],
        zero_division=0
    ))