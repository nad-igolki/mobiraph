import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import joblib
import numpy as np



HIERARCHY_ROOTS = [
        "root",
        "Class I (Retrotransposons)",
        "Class II (DNA transposons)",
        "Class I (Retrotransposons)\tLTR Retrotransposon",
        "Class I (Retrotransposons)\tNon-LTR Retrotransposon",
    ]
results_dir = "/Users/nad/mobiraph/data/n13_repbase_processed_wo_bad_sf/train_logits"

# general_df = pd.DataFrame()
# for hierarchy_root in HIERARCHY_ROOTS:
#     df_hyena = pd.read_csv(f"{results_dir}/{hierarchy_root}/hyena_catboost.csv")
#     df_kmer = pd.read_csv(f"{results_dir}/{hierarchy_root}/kmer_cnn.csv")
#     df_gat = pd.read_csv(f"{results_dir}/{hierarchy_root}/gat.csv")
#     df_hyena = df_hyena.drop('y_pred', axis=1)
#     # df_hyena = df_hyena.drop('y_true', axis=1)
#     df_kmer = df_kmer.drop('y_pred', axis=1)
#     # df_kmer = df_kmer.drop('y_true', axis=1)
#     df_gat = df_gat.drop('y_pred', axis=1)
#     # df_gat = df_gat.drop('y_true', axis=1)
#     df_both = pd.merge(df_hyena, df_kmer, on='name', how='inner')
#     df_both = pd.merge(df_both, df_gat, on='name', how='inner')
#     if general_df.empty:
#         general_df = df_both
#     else:
#         general_df_copy = pd.merge(general_df, df_both, on='name', how='inner', suffixes=('_gen', '_both'))
#         general_df = general_df_copy
#     print(general_df.shape)

general_df = pd.DataFrame()

for hierarchy_root in HIERARCHY_ROOTS:
    df_hyena = pd.read_csv(f"{results_dir}/{hierarchy_root}/hyena_catboost.csv")
    df_kmer = pd.read_csv(f"{results_dir}/{hierarchy_root}/kmer_cnn.csv")
    # df_gat = pd.read_csv(f"{results_dir}/{hierarchy_root}/gat.csv")
    df_20 = pd.read_csv(f"{results_dir}/{hierarchy_root}/kmer_cnn_20.csv")
    df_30 = pd.read_csv(f"{results_dir}/{hierarchy_root}/kmer_cnn_30.csv")
    df_123 = pd.read_csv(f"{results_dir}/{hierarchy_root}/kmer_cnn_123.csv")

    df_hyena = df_hyena.drop('y_pred', axis=1)
    df_kmer = df_kmer.drop('y_pred', axis=1)
    df_20 = df_20.drop('y_pred', axis=1)
    df_30 = df_30.drop('y_pred', axis=1)
    df_123 = df_123.drop('y_pred', axis=1)
    # df_gat = df_gat.drop('y_pred', axis=1)

    df_hyena = df_hyena.rename(columns={
        col: f"{col}_hyena_{hierarchy_root}" for col in df_hyena.columns if col != 'name'
    })
    df_kmer = df_kmer.rename(columns={
        col: f"{col}_kmer_{hierarchy_root}" for col in df_kmer.columns if col != 'name'
    })
    # df_gat = df_gat.rename(columns={
    #     col: f"{col}_gat_{hierarchy_root}" for col in df_gat.columns if col != 'name'
    # })
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
    # df_both = pd.merge(df_both, df_gat, on='name', how='inner')
    df_both = pd.merge(df_both, df_20, on='name', how='inner')
    df_both = pd.merge(df_both, df_30, on='name', how='inner')
    df_both = pd.merge(df_both, df_123, on='name', how='inner')

    if general_df.empty:
        general_df = df_both
    else:
        general_df = pd.merge(general_df, df_both, on='name', how='inner')

    print(general_df.shape)

# general_df.to_csv('output.csv', index=False)

import json
with open("/Users/nad/mobiraph/data/n13_repbase_processed/metadata_03.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(type(metadata))

y_true = []
for value in general_df['name']:
    # if value not in metadata:
    #     y_true.append(np.nan)
    #     continue
    # if 'superfamily' not in metadata[value]:
    #     y_true.append(np.nan)
    #     continue
    # if metadata[value]['superfamily'] in ["Academ", "DNA transposon_other", "Kolobok", "Troyka", "Non-LTR Retrotransposon_other", "piggyBac"]:
    #     y_true.append(np.nan)
    # else:
    y_true.append(metadata[value]['superfamily'])

general_df['y_true'] = y_true
general_df['y_true'] = general_df['y_true'].astype(str)
general_df = general_df[general_df['y_true'] != 'nan']

general_df = general_df.dropna()
# general_df.to_csv('output.csv', index=False)

save_path = f"/Users/nad/mobiraph/data/n13_repbase_processed_wo_bad_sf/ensemble_model/XGBClassifier.pkl"


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

    # from sklearn.ensemble import HistGradientBoostingClassifier
    # from sklearn.utils.class_weight import compute_sample_weight
    #
    # sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    # model = HistGradientBoostingClassifier(
    #     max_iter=300,
    #     learning_rate=0.05,
    #     max_depth=8,
    #     random_state=42
    # )
    #
    # model.fit(X_train, y_train, sample_weight=sample_weight)
    # model.fit(X_train, y_train)
    # from sklearn.tree import DecisionTreeClassifier
    # model = DecisionTreeClassifier(
    #     class_weight="balanced",  # балансировка классов
    #     max_depth=25,  # можно подбирать
    #     min_samples_split=10,
    #     min_samples_leaf=5,
    #     random_state=random_state
    # )
    #
    # model.fit(X_train, y_train)
    # from sklearn.ensemble import RandomForestClassifier
    # model = RandomForestClassifier(
    #     n_estimators=500,
    #     max_depth=16,
    #     min_samples_split=10,
    #     min_samples_leaf=3,
    #     max_features=0.5,
    #     class_weight="balanced_subsample",
    #     random_state=42,
    #     n_jobs=-1
    # )
    #
    # model.fit(X_train, y_train)

    # from catboost import CatBoostClassifier
    # from sklearn.utils.class_weight import compute_class_weight
    # import numpy as np
    #
    # classes = np.unique(y_train)
    # weights = compute_class_weight(
    #     class_weight="balanced",
    #     classes=classes,
    #     y=y_train
    # )
    # class_weights = dict(zip(classes, weights))
    #
    # model = CatBoostClassifier(
    #     loss_function="MultiClass",
    #     iterations=1000,
    #     learning_rate=0.05,
    #     depth=6,
    #     l2_leaf_reg=3,
    #     random_seed=random_state,
    #     verbose=100,
    #     class_weights=class_weights
    # )
    #
    # model.fit(X_train, y_train)
    sample_weight = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(le.classes_),
        n_estimators=2000,  # верхняя граница
        max_depth=4,  # было 6
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

    # y_pred_enc = model.predict(X_test)
    # y_pred = le.inverse_transform(y_pred_enc)
    # y_test_labels = le.inverse_transform(y_test)
    #
    # print(classification_report(y_test_labels, y_pred, labels=list(le.classes_), zero_division=0))

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
