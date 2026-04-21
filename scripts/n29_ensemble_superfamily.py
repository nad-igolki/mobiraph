import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import numpy as np



HIERARCHY_ROOTS = [
        "root",
        "Class I (Retrotransposons)",
        "Class II (DNA transposons)",
        "Class I (Retrotransposons)\tLTR Retrotransposon",
        "Class I (Retrotransposons)\tNon-LTR Retrotransposon",
    ]
results_dir = "/Users/nad/mobiraph/data/n34_train_results_new"

general_df = pd.DataFrame()
for hierarchy_root in HIERARCHY_ROOTS:
    df_hyena = pd.read_csv(f"{results_dir}/{hierarchy_root}/hyena_transformer.csv")
    df_kmer = pd.read_csv(f"{results_dir}/{hierarchy_root}/kmer_cnn.csv")
    df_hyena = df_hyena.drop('y_pred', axis=1)
    df_hyena = df_hyena.drop('y_true', axis=1)
    df_kmer = df_kmer.drop('y_pred', axis=1)
    df_kmer = df_kmer.drop('y_true', axis=1)
    df_both = pd.merge(df_hyena, df_kmer, on='name', how='left')
    if general_df.empty:
        general_df = df_both
    else:
        general_df_copy = pd.merge(general_df, df_both, on='name', how='left')
        general_df = general_df_copy
    print(general_df.shape)

import json
with open("/Users/nad/mobiraph/data/n13_repbase_processed/metadata_03.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(type(metadata))

y_true = []
for value in general_df['name']:
    if 'superfamily' not in metadata[value]:
        y_true.append(np.nan)
        continue
    if metadata[value]['superfamily'] in ["Academ", "DNA transposon_other", "Kolobok", "Troyka", "Non-LTR Retrotransposon_other", "piggyBac"]:
        y_true.append(np.nan)
    else:
        y_true.append(metadata[value]['superfamily'])

general_df['y_true'] = y_true
general_df['y_true'] = general_df['y_true'].astype(str)
general_df = general_df[general_df['y_true'] != 'nan']

save_path = f"/Users/nad/mobiraph/data/n30_ensemble_superfamily_new/train.pkl"


def train(
    df: pd.DataFrame,
    save_path: str,
    target_col: str = "y_true",
    test_size: float = 0.2,
    random_state: int = 42,
):
    feature_cols = [c for c in df.columns if c not in ["name", "y_true"]]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df[target_col].astype(str)

    le = LabelEncoder()
    le.fit(y.unique())

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


    joblib.dump(bundle, save_path)
    print(f"Модель сохранена в: {save_path}")

    return bundle


bundle = train(
    df=general_df,
    target_col="y_true",
    save_path=save_path
)
