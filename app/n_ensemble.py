import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import joblib
import numpy as np


def ensemble_train(results_dir: str, checkpoints_dir: str, metadata_path: str):
    HIERARCHY_ROOTS = [
            "root",
            "Class I (Retrotransposons)",
            "Class II (DNA transposons)",
            "Class I (Retrotransposons)\tLTR Retrotransposon",
            "Class I (Retrotransposons)\tNon-LTR Retrotransposon",
        ]

    general_df = pd.DataFrame()

    for hierarchy_root in HIERARCHY_ROOTS:
        df_hyena = pd.read_csv(f"{results_dir}/{hierarchy_root}/hyena_catboost.csv")
        df_kmer = pd.read_csv(f"{results_dir}/{hierarchy_root}/kmer_cnn.csv")
        df_gat = pd.read_csv(f"{results_dir}/{hierarchy_root}/gat.csv")
        df_20 = pd.read_csv(f"{results_dir}/{hierarchy_root}/kmer_cnn_20.csv")
        df_30 = pd.read_csv(f"{results_dir}/{hierarchy_root}/kmer_cnn_30.csv")
        df_123 = pd.read_csv(f"{results_dir}/{hierarchy_root}/kmer_cnn_123.csv")

        df_hyena = df_hyena.drop('y_pred', axis=1)
        df_kmer = df_kmer.drop('y_pred', axis=1)
        df_20 = df_20.drop('y_pred', axis=1)
        df_30 = df_30.drop('y_pred', axis=1)
        df_123 = df_123.drop('y_pred', axis=1)
        df_gat = df_gat.drop('y_pred', axis=1)

        df_hyena = df_hyena.rename(columns={
            col: f"{col}_hyena_{hierarchy_root}" for col in df_hyena.columns if col != 'name'
        })
        df_kmer = df_kmer.rename(columns={
            col: f"{col}_kmer_{hierarchy_root}" for col in df_kmer.columns if col != 'name'
        })
        df_gat = df_gat.rename(columns={
            col: f"{col}_gat_{hierarchy_root}" for col in df_gat.columns if col != 'name'
        })
        df_20 = df_20.rename(columns={
            col: f"{col}_20_{hierarchy_root}" for col in df_20.columns if col != 'name'
        })
        df_30 = df_30.rename(columns={
            col: f"{col}_30_{hierarchy_root}" for col in df_30.columns if col != 'name'
        })
        df_123 = df_123.rename(columns={
            col: f"{col}_30_{hierarchy_root}" for col in df_123.columns if col != 'name'
        })

        df_both = pd.merge(df_hyena, df_kmer, on='name', how='inner')
        df_both = pd.merge(df_both, df_gat, on='name', how='inner')
        df_both = pd.merge(df_both, df_20, on='name', how='inner')
        df_both = pd.merge(df_both, df_30, on='name', how='inner')
        df_both = pd.merge(df_both, df_123, on='name', how='inner')

        if general_df.empty:
            general_df = df_both
        else:
            general_df = pd.merge(general_df, df_both, on='name', how='inner')

        print(general_df.shape)
    import json
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    y_true = []
    for value in general_df['name']:
        y_true.append(metadata[value]['superfamily'])

    general_df['y_true'] = y_true
    general_df['y_true'] = general_df['y_true'].astype(str)
    general_df = general_df[general_df['y_true'] != 'nan']

    general_df = general_df.dropna()

    save_path = f"{checkpoints_dir}/ensemble_model/XGBClassifier.pkl"


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

        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y_train
        )

        model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(le.classes_),
            n_estimators=2000,
            max_depth=4,
            min_child_weight=5,
            gamma=0.2,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=3.0,
            reg_alpha=0.5,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="mlogloss"
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        fi = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        print("\nTop features:")
        print(fi.head(20))

        y_pred_enc = model.predict(X_test)
        y_pred_enc = np.asarray(y_pred_enc).ravel().astype(int)

        y_test = np.asarray(y_test).ravel().astype(int)

        y_pred = le.inverse_transform(y_pred_enc)
        y_test_labels = le.inverse_transform(y_test)

        print(classification_report(
            y_test_labels,
            y_pred,
            labels=le.classes_,
            zero_division=0
        ))

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


import os
import joblib
import pandas as pd


def ensemble_test(
    results_dir: str,
    models_path: str,
    output_file: str
):
    HIERARCHY_ROOTS = [
        "root",
        "Class I (Retrotransposons)",
        "Class II (DNA transposons)",
        "Class I (Retrotransposons)\tLTR Retrotransposon",
        "Class I (Retrotransposons)\tNon-LTR Retrotransposon",
    ]

    model_files = {
        "hyena": "hyena_catboost.csv",
        "kmer": "kmer_cnn.csv",
        "gat": "gat.csv",
        "20": "kmer_cnn_20.csv",
        "30": "kmer_cnn_30.csv",
        "123": "kmer_cnn_123.csv",
    }

    general_df = None

    for hierarchy_root in HIERARCHY_ROOTS:
        dfs = []

        for model_name, filename in model_files.items():
            path = os.path.join(results_dir, hierarchy_root, filename)

            df = pd.read_csv(path)

            # y_true на тесте нет, y_pred от базовой модели не нужен как отдельный ответ
            df = df.drop(columns=["y_true", "y_pred"], errors="ignore")

            # переименовываем все признаки, кроме name
            df = df.rename(columns={
                col: f"{col}_{model_name}_{hierarchy_root}"
                for col in df.columns
                if col != "name"
            })

            dfs.append(df)

        df_merged = dfs[0]
        for df in dfs[1:]:
            df_merged = pd.merge(df_merged, df, on="name", how="inner")

        if general_df is None:
            general_df = df_merged
        else:
            general_df = pd.merge(general_df, df_merged, on="name", how="inner")

        print(f"{hierarchy_root}: {general_df.shape}")

    model_path = os.path.join(models_path, "ensemble_model", "XGBClassifier.pkl")
    bundle = joblib.load(model_path)

    feature_cols = bundle["feature_cols"]
    label_encoder = bundle["label_encoder"]
    model = bundle["model"]

    # добавляем отсутствующие признаки нулями,
    # чтобы структура совпадала с обучением ensemble-модели
    for col in feature_cols:
        if col not in general_df.columns:
            general_df[col] = 0.0

    X = (
        general_df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    y_pred = label_encoder.inverse_transform(model.predict(X))

    res = general_df[["name"]].copy()
    res["superfamily_pred"] = y_pred

    superfamily_to_order = {
        # Class II — DNA transposons
        "Mariner/Tc1": "DNA transposon",
        "hAT": "DNA transposon",
        "MuDR": "DNA transposon",
        "EnSpm/CACTA": "DNA transposon",
        "piggyBac": "DNA transposon",
        "Harbinger": "DNA transposon",
        "Helitron": "DNA transposon",
        "Kolobok": "DNA transposon",
        "Academ": "DNA transposon",
        "DNA transposon_other": "DNA transposon",

        # Class I — LTR retrotransposons
        "Gypsy": "LTR Retrotransposon",
        "Copia": "LTR Retrotransposon",
        "BEL": "LTR Retrotransposon",
        "DIRS": "LTR Retrotransposon",
        "Troyka": "LTR Retrotransposon",

        # Class I — Non-LTR retrotransposons
        "SINE": "Non-LTR Retrotransposon",
        "L1": "Non-LTR Retrotransposon",
        "RTE": "Non-LTR Retrotransposon",
        "CR1": "Non-LTR Retrotransposon",
        "Tx1": "Non-LTR Retrotransposon",
        "RTEX": "Non-LTR Retrotransposon",
        "Tad1": "Non-LTR Retrotransposon",
        "Non-LTR Retrotransposon_other": "Non-LTR Retrotransposon",
    }

    superfamily_to_class = {
        # Class II — DNA transposons
        "Mariner/Tc1": "Class II (DNA transposons)",
        "hAT": "Class II (DNA transposons)",
        "MuDR": "Class II (DNA transposons)",
        "EnSpm/CACTA": "Class II (DNA transposons)",
        "piggyBac": "Class II (DNA transposons)",
        "Harbinger": "Class II (DNA transposons)",
        "Helitron": "Class II (DNA transposons)",
        "Kolobok": "Class II (DNA transposons)",
        "Academ": "Class II (DNA transposons)",
        "DNA transposon_other": "Class II (DNA transposons)",

        # Class I — LTR retrotransposons
        "Gypsy": "Class I (Retrotransposons)",
        "Copia": "Class I (Retrotransposons)",
        "BEL": "Class I (Retrotransposons)",
        "DIRS": "Class I (Retrotransposons)",
        "Troyka": "Class I (Retrotransposons)",

        # Class I — Non-LTR retrotransposons
        "SINE": "Class I (Retrotransposons)",
        "L1": "Class I (Retrotransposons)",
        "RTE": "Class I (Retrotransposons)",
        "CR1": "Class I (Retrotransposons)",
        "Tx1": "Class I (Retrotransposons)",
        "RTEX": "Class I (Retrotransposons)",
        "Tad1": "Class I (Retrotransposons)",
        "Non-LTR Retrotransposon_other": "Class I (Retrotransposons)",
    }

    res["order_pred"] = res["superfamily_pred"].map(superfamily_to_order)
    res["class_pred"] = res["superfamily_pred"].map(superfamily_to_class)
    unknown_mask = res["order_pred"].isna() | res["class_pred"].isna()
    if unknown_mask.any():
        unknown_values = res.loc[unknown_mask, "superfamily_pred"].unique()
        print("Warning: unknown predicted superfamilies:", unknown_values)

    res.to_csv(output_file, index=False)

    print(f"Saved predictions to: {output_file}")
    print(res.head())

    return res