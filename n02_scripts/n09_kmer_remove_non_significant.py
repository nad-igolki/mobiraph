import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import gc


import config


def kmer_standardize_and_filter(
    dataset_dir: str,
    k: int,
    number_of_columns: int = 50,
    mode: str | None = None,
    seed: int | None = None,
):
    if mode is not None:
        path = os.path.join(dataset_dir, f'{k}_{mode}_{seed}.csv')
        save_path = os.path.join(dataset_dir, f'{k}_{mode}_{seed}_trimmed.csv')
    else:
        path = os.path.join(dataset_dir, f'{k}.csv')
        save_path = os.path.join(dataset_dir, f'{k}_trimmed.csv')

    df = pd.read_csv(path)

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        df.to_csv(save_path, index=False)
        print("No embedding columns found, saved original file.")
        return

    nonzero_counts = (df[emb_cols] != 0).sum(axis=0)

    # выбираем number_of_columns колонок с максимальным количеством ненулевых элементов
    top50_cols = nonzero_counts.sort_values(ascending=False).head(number_of_columns).index

    # оставляем только эти колонки в df
    df_top50 = df[top50_cols]

    df_top50.to_csv(save_path, index=False)

    print(
        f"Removed {len(df.columns) - len(df_top50.columns)} columns "
        f"Final shape: {df_top50.shape}"
    )

if __name__ == '__main__':
    ks = [4, 5, 6, 7]

    for i in tqdm(range(len(ks))):
        kmer_standardize_and_filter(config.DIR_KMER_DATASETS, ks[i], mode=0, seed=1)
    for i in tqdm(range(len(ks))):
        kmer_standardize_and_filter(config.DIR_INCEST_MANY, ks[i])
