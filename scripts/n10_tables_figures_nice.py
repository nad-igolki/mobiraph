import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path


def plot_classification_report(
    y_true,
    y_pred,
    title="Classification Report",
    save_path: Path | None = None
):
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose().round(2)

    df_for_plot = df.drop(columns=["support"], errors="ignore")

    plt.figure(figsize=(8, len(df_for_plot) * 0.6))
    sns.heatmap(df_for_plot, annot=True, cmap="PuBu", cbar=False, fmt=".2f")
    plt.title(title, fontsize=14)
    plt.yticks(rotation=0)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Сохранено: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    title="Confusion Matrix",
    save_path: Path | None = None,
    normalize: bool = False
):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = cm.round(2)

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df_cm,
        annot=True,
        cmap="PuBu",
        cbar=False,
        fmt=".2f" if normalize else "d"
    )

    plt.title(title, fontsize=14)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha="right")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Сохранено: {save_path}")
    else:
        plt.show()

    plt.close()
