import pandas as pd
import os
from tqdm import tqdm


import config


def kmer_remove_non_significant(dataset_dir: str, k: int, threshold: int, mode=None, seed=None):
    if mode is not None:
        path = os.path.join(dataset_dir, f'{k}_{mode}_{seed}.csv')
        save_path = os.path.join(dataset_dir, f'{k}_{mode}_{seed}_trimmed.csv')
    else:
        path = os.path.join(dataset_dir, f'{k}.csv')
        save_path = os.path.join(dataset_dir, f'{k}_trimmed.csv')
    df = pd.read_csv(path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    # nonzero_counts = (df[emb_cols] != 0).sum().sort_values(ascending=False)
    # mean_nonzero = nonzero_counts.mean()
    # print(mean_nonzero, df.shape)

    cols_to_remove = []
    for c in tqdm(emb_cols):
        nonzero_count = (df[c] != 0).sum()
        if nonzero_count <= threshold:
            cols_to_remove.append(c)
    df = df.drop(columns=cols_to_remove)

    df.to_csv(save_path, index=False)
    print(f'Removed {len(cols_to_remove)} non-significant entries')


if __name__ == '__main__':
    ks = [4, 5, 6, 7]
    # thresholds = [100, 60, 30, 10]
    # for i in tqdm(range(len(ks))):
    #     kmer_remove_non_significant(config.DIR_INCEST_MANY, ks[i], thresholds[i])

    thresholds = [3000, 2900, 1800, 1000]
    for i in tqdm(range(len(ks))):
        kmer_remove_non_significant(config.DIR_KMER_DATASETS, ks[i], thresholds[i], 0, 1)
