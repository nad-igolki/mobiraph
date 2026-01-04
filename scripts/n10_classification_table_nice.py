import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report


def plot_classification_report(y_true, y_pred, title="Classification Report"):
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose().round(2)

    df_for_plot = df.drop(columns=["support"], errors="ignore")

    plt.figure(figsize=(8, len(df_for_plot) * 0.6))
    sns.heatmap(df_for_plot, annot=True, cmap="PuBu", cbar=False, fmt=".2f")
    plt.title(title, fontsize=14)
    plt.yticks(rotation=0)
    plt.show()
