import os, warnings, json
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scripts.n12_cnn_model import CNNClassifierModel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)
import matplotlib.pyplot as plt


os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
os.chdir(config.DIR_ROOT)

REPO_ROOT = Path(config.DIR_ROOT)
mlflow.set_tracking_uri((REPO_ROOT / "mlruns").as_uri())
mlflow.set_experiment("cnn-7mer")

embeddings_7mer_path = os.path.join(config.DIR_INCEST_MANY, '7.csv')
types_7mer_path = os.path.join(config.DIR_INCEST_MANY, 'repbase_filtered.csv')

with mlflow.start_run(run_name="cnn_7mer"):

    mlflow.log_param("embeddings_path", embeddings_7mer_path)
    mlflow.log_param("types_path", types_7mer_path)
    mlflow.log_param("epochs", 20)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    embeddings_7mer = pd.read_csv(embeddings_7mer_path)
    types_7mer = pd.read_csv(types_7mer_path, sep=',')


    df = types_7mer.query("Good == 1")[['name', 'MainType']].merge(
        embeddings_7mer, on='name', how='inner'
    )

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X = df[emb_cols].to_numpy(dtype=np.float32)

    le = LabelEncoder()
    y = le.fit_transform(df['MainType'].astype(str))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.log_param("n_total", int(len(df)))
    mlflow.log_param("n_train", int(len(X_train)))
    mlflow.log_param("n_val", int(len(X_val)))
    mlflow.log_param("input_dim", int(X_train.shape[1]))
    mlflow.log_param("class_num", int(len(le.classes_)))

    # classes as artifact
    mlflow.log_dict(
        {"classes": le.classes_.tolist()},
        artifact_file="metadata/label_classes.json"
    )

    model = CNNClassifierModel(input_dim=X_train.shape[1], class_num=len(le.classes_))
    history = model.train(X_train, y_train, X_val=X_val, y_val=y_val, epochs=20, batch_size=32)

    import mlflow.keras

    mlflow.keras.log_model(
        model.model,
        artifact_path="model"
    )

    # log history if possible
    if isinstance(history, dict):
        if "val_loss" in history:
            for epoch, v in enumerate(history["val_loss"]):
                mlflow.log_metric("val_loss", float(v), step=epoch)
        if "train_loss" in history:
            for epoch, v in enumerate(history["train_loss"]):
                mlflow.log_metric("train_loss", float(v), step=epoch)
        if "val_acc" in history:
            for epoch, v in enumerate(history["val_acc"]):
                mlflow.log_metric("val_acc", float(v), step=epoch)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    precision = precision_score(
        y_val, y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_val, y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_val, y_pred,
        average="macro",
        zero_division=0
    )

    mcc = matthews_corrcoef(y_val, y_pred)

    cm = confusion_matrix(y_val, y_pred)

    specificities = []

    for i in range(len(cm)):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)

        if (TN + FP) > 0:
            specificities.append(TN / (TN + FP))

    specificity = np.mean(specificities)

    gmean = np.sqrt(recall * specificity)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision_macro", precision)
    mlflow.log_metric("recall_macro", recall)
    mlflow.log_metric("f1_macro", f1)
    mlflow.log_metric("specificity_macro", specificity)
    mlflow.log_metric("gmean_macro", gmean)
    mlflow.log_metric("mcc", mcc)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    plt.colorbar(ax.images[0], ax=ax)
    fig.tight_layout()

    mlflow.log_figure(fig, "figures/confusion_matrix.pdf")
    plt.close(fig)

