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
results_dir = "/Users/nad/mobiraph/data/n22_test_results"

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

model_path = f"/Users/nad/mobiraph/data/n30_ensemble_superfamily/train.pkl"
bundle = joblib.load(model_path)

def predict_on_new_data(bundle, df):
    feature_cols = bundle["feature_cols"]
    print(feature_cols)
    le = bundle["label_encoder"]
    model = bundle["model"]

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    pred = le.inverse_transform(model.predict(X))

    out = df[["name"]].copy()
    out["prediction"] = pred

    return out

preds = predict_on_new_data(
        bundle=bundle,
        df = general_df
    )

res = general_df[["name", "y_true"]].merge(preds, on="name", how="left")

print(classification_report(
    res["y_true"].astype(str),
    res["prediction"].astype(str),
    labels=bundle["classes"],
    zero_division=0
))


for cls in res['y_true'].unique():
    mask = res['y_true'] == cls
    acc = (res.loc[mask, 'y_true'] == res.loc[mask, 'prediction']).mean()
    print(f"{cls}: {acc:.3f}")


y_class_true = []
for value in res['name']:
    y_class_true.append(metadata[value]['class'])
res['y_class_true'] = y_class_true



superfamily_to_class = {
    # Class 2 — DNA transposons
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

    # Class 1 — LTR retrotransposons
    "Gypsy": "Class I (Retrotransposons)",
    "Copia": "Class I (Retrotransposons)",
    "BEL": "Class I (Retrotransposons)",
    "DIRS": "Class I (Retrotransposons)",
    "Troyka": "Class I (Retrotransposons)",

    # Class 1 — Non-LTR retrotransposons
    "SINE": "Class I (Retrotransposons)",
    "L1": "Class I (Retrotransposons)",
    "RTE": "Class I (Retrotransposons)",
    "CR1": "Class I (Retrotransposons)",
    "Tx1": "Class I (Retrotransposons)",
    "RTEX": "Class I (Retrotransposons)",
    "Tad1": "Class I (Retrotransposons)",
    "Non-LTR Retrotransposon_other": "Class I (Retrotransposons)"
}

y_class_pred = []
for value in res['name']:
    y_class_pred.append(superfamily_to_class[res.loc[res['name'] == value, 'prediction'].iloc[0]])
res['y_class_pred'] = y_class_pred


for cls in res['y_class_true'].unique():
    mask = res['y_class_true'] == cls
    acc = (res.loc[mask, 'y_class_true'] == res.loc[mask, 'y_class_pred']).mean()
    print(f"{cls}: {acc:.3f}")





def extract_sequences(fasta_file, headers_file, output_file):
    # Загружаем заголовки
    with open(headers_file) as f:
        target_headers = set(line.strip() for line in f if line.strip())

    # Парсим FASTA и пишем совпадения
    with open(fasta_file) as fin, open(output_file, "w") as fout:
        write = False
        for line in fin:
            if line.startswith(">"):
                header = line[1:].strip()
                write = header in target_headers
            if write:
                fout.write(line)

extract_sequences("/Users/nad/mobiraph/data/n13_repbase_processed/all_sequences_filtered_02_ltr_correction.fasta", "/Users/nad/mobiraph/data/n13_repbase_processed/id_test.txt", "/Users/nad/mobiraph/data/n13_repbase_processed/all_sequences_filtered_test.fasta")
