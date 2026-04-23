import pandas as pd
import numpy as np
import json

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# === CONFIG ===
N_RUNS = 100

HIERARCHY_ROOTS = [
    "root",
    "Class I (Retrotransposons)",
    "Class II (DNA transposons)",
    "Class I (Retrotransposons)\tLTR Retrotransposon",
    "Class I (Retrotransposons)\tNon-LTR Retrotransposon",
]

train_dir = "/Users/nad/mobiraph/data/n35_train_results"
test_dir  = "/Users/nad/mobiraph/data/n37_test_results"
metadata_path = "/Users/nad/mobiraph/data/n13_repbase_processed/metadata_03.json"

# === BUILD DF ===
def build_df(results_dir):
    general_df = pd.DataFrame()

    for root in HIERARCHY_ROOTS:
        df_hyena = pd.read_csv(f"{results_dir}/{root}/hyena_catboost.csv")
        df_kmer  = pd.read_csv(f"{results_dir}/{root}/kmer_cnn.csv")

        df_hyena = df_hyena.drop(columns=['y_pred','y_true'], errors='ignore')
        df_kmer  = df_kmer.drop(columns=['y_pred','y_true'], errors='ignore')

        df = pd.merge(df_hyena, df_kmer, on='name', how='left')

        if general_df.empty:
            general_df = df
        else:
            general_df = pd.merge(general_df, df, on='name', how='left')

    return general_df

# === LOAD DATA ===
train_df = build_df(train_dir)
test_df  = build_df(test_dir)

with open(metadata_path) as f:
    metadata = json.load(f)

def add_target(df):
    y = []
    for name in df['name']:
        if name in metadata and 'superfamily' in metadata[name]:
            sf = metadata[name]['superfamily']
            if sf not in ["Academ","DNA transposon_other","Kolobok",
                          "Troyka","Non-LTR Retrotransposon_other","piggyBac"]:
                y.append(sf)
            else:
                y.append(np.nan)
        else:
            y.append(np.nan)
    df['y_true'] = y
    return df.dropna()

train_df = add_target(train_df)
test_df  = add_target(test_df)

# === FEATURES ===
feature_cols = [c for c in train_df.columns if c not in ["name","y_true"]]

X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
y_train = train_df['y_true'].astype(str)

X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
y_test = test_df['y_true'].astype(str)

# === LABEL ENCODER ===
le = LabelEncoder()
le.fit(y_train)

y_train_enc = le.transform(y_train)
y_test_enc  = le.transform(y_test)

# === STORAGE ===
metrics = {
    "precision": [],
    "recall": [],
    "f1": [],
    "specificity": []
}

# === RUNS ===
for run in range(N_RUNS):
    print(f"run={run}")

    sample_weight = compute_sample_weight("balanced", y_train_enc)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(le.classes_),
        n_estimators=2000,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=run,
        n_jobs=-1,
        eval_metric="mlogloss"
    )

    model.fit(X_train, y_train_enc, sample_weight=sample_weight, verbose=False)

    y_pred = model.predict(X_test)

    # === weighted metrics ===
    p, r, f1, _ = precision_recall_fscore_support(
        y_test_enc, y_pred, average="weighted", zero_division=0
    )

    # === specificity ===
    cm = confusion_matrix(y_test_enc, y_pred)
    spec_list = []

    for i in range(len(cm)):
        tn = np.sum(cm) - (np.sum(cm[i,:]) + np.sum(cm[:,i]) - cm[i,i])
        fp = np.sum(cm[:,i]) - cm[i,i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        spec_list.append(spec)

    specificity = np.average(spec_list)

    metrics["precision"].append(p)
    metrics["recall"].append(r)
    metrics["f1"].append(f1)
    metrics["specificity"].append(specificity)

# === CI ===
def ci(arr):
    arr = np.array(arr)
    mean = arr.mean()
    std = arr.std(ddof=1)
    margin = 1.96 * std / np.sqrt(len(arr))
    return mean, mean - margin, mean + margin

# === OUTPUT ===
for k, v in metrics.items():
    mean, low, high = ci(v)
    print(f"{k}:")
    print(f"mean={mean:.4f}, CI95=({low:.4f}, {high:.4f})")

# === SAVE PER-RUN METRICS ===
metrics_df = pd.DataFrame(metrics)
metrics_df.index.name = "run"
metrics_df.to_csv("/Users/nad/mobiraph/figures/metrics_per_run.csv", index=True)

# === SAVE SUMMARY WITH CI ===
summary = []
for k, v in metrics.items():
    mean, low, high = ci(v)
    summary.append({
        "metric": k,
        "mean": mean,
        "ci95_low": low,
        "ci95_high": high
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("/Users/nad/mobiraph/figures/metrics_summary.csv", index=False)

print("Saved: metrics_per_run.csv, metrics_summary.csv")