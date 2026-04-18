import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


FILE1 = "/Users/nad/mobiraph/data/n22_test_results/root/hyena_transformer.csv"
FILE2 = "/Users/nad/mobiraph/data/n22_test_results/root/kmer_cnn.csv"

ID_COL = "name"
TRUE_COL = "y_true"
PRED_COL = "y_pred"
CLASS_COLS = [
    "Class I (Retrotransposons)","Class II (DNA transposons)"
]


def load_model_file(path, prefix, id_col=ID_COL, true_col=TRUE_COL, pred_col=PRED_COL, class_cols=CLASS_COLS):
    df = pd.read_csv(path)

    required = [id_col, true_col, pred_col] + class_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    df = df[required].copy()
    rename_map = {c: f"{prefix}__{c}" for c in class_cols + [pred_col, true_col]}
    return df.rename(columns=rename_map)


def merge_model_outputs(file_paths, class_cols=CLASS_COLS):
    dfs = []
    prefixes = []

    for i, path in enumerate(file_paths, start=1):
        prefix = f"m{i}"
        prefixes.append(prefix)
        dfs.append(load_model_file(path, prefix, class_cols=class_cols))

    df = dfs[0]
    for other in dfs[1:]:
        df = df.merge(other, on=ID_COL, how="inner")

    first_true = f"{prefixes[0]}__{TRUE_COL}"
    for prefix in prefixes[1:]:
        col = f"{prefix}__{TRUE_COL}"
        if not (df[first_true] == df[col]).all():
            bad = df.loc[df[first_true] != df[col], [ID_COL, first_true, col]]
            raise ValueError(f"y_true mismatch between files. Example:\n{bad.head()}")

    df[TRUE_COL] = df[first_true]
    return df, prefixes


def build_stack_features(df, prefixes, class_cols=CLASS_COLS):
    parts = []
    for prefix in prefixes:
        cols = [f"{prefix}__{c}" for c in class_cols]
        parts.append(df[cols].values)
    return np.hstack(parts)


def evaluate(y_true, y_pred, title="Stacking"):
    print(f"\n===== {title} =====")
    print(f"Accuracy    : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1    : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Weighted F1 : {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred))


def train_stacking_logreg(
    file_paths,
    class_cols=CLASS_COLS,
    output_path="/Users/nad/mobiraph/data/n24_ensemble_models/stacking_predictions_root.csv",
    model_path="/Users/nad/mobiraph/data/n24_ensemble_models/stacking_model_root.pkl"
):
    df, prefixes = merge_model_outputs(file_paths, class_cols=class_cols)

    X = build_stack_features(df, prefixes, class_cols=class_cols)
    y = df[TRUE_COL].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", LogisticRegression(max_iter=5000))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pred_enc = cross_val_predict(model, X, y_enc, cv=cv, method="predict")
    pred = le.inverse_transform(pred_enc)

    proba = cross_val_predict(model, X, y_enc, cv=cv, method="predict_proba")

    evaluate(y, pred, "Stacking LogisticRegression (5-fold CV)")

    # финальное обучение на всех данных
    model.fit(X, y_enc)

    # --- СОХРАНЕНИЕ МОДЕЛИ ---
    joblib.dump({
        "model": model,
        "label_encoder": le,
        "class_cols": class_cols,
        "prefixes": prefixes
    }, model_path)
    print(f"\nSaved model to: {model_path}")
    # ------------------------

    out = df[[ID_COL, TRUE_COL]].copy()
    for prefix in prefixes:
        out[f"pred_{prefix}"] = df[f"{prefix}__{PRED_COL}"]
    out["pred_stacking"] = pred
    out["stack_correct"] = out["pred_stacking"] == out[TRUE_COL]

    for i, cls in enumerate(le.classes_):
        out[f"stack_proba__{cls}"] = proba[:, i]

    out.to_csv(output_path, index=False)
    print(f"Saved predictions: {output_path}")

    return model, le, out


if __name__ == "__main__":
    file_paths = [FILE1, FILE2]
    model, label_encoder, results = train_stacking_logreg(file_paths)