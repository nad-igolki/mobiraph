import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import gc


import config


def kmer_standardize_and_filter(
    dataset_dir: str,
    k: int,
    threshold: float = 0.01,
    mode: str | None = None,
    seed: int | None = None,
    use_abs_max: bool = True,
    zero_only_drop: bool = True
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

    df[emb_cols] = df[emb_cols].astype(np.float32)
    if df[emb_cols].isna().any().any():
        df[emb_cols] = df[emb_cols].fillna(0.0)

    # Сразу выкинем колонки, где все нули
    if zero_only_drop:
        nonzero_mask = (df[emb_cols] != 0).any(axis=0).values
        kept_zero_drop = np.array(emb_cols)[nonzero_mask].tolist()
        dropped_zero = np.array(emb_cols)[~nonzero_mask].tolist()
        if dropped_zero:
            df = df.drop(columns=dropped_zero)
            emb_cols = kept_zero_drop


    if use_abs_max:
        col_max = np.nanmax(np.abs(df[emb_cols]), axis=0)
    else:
        col_max = np.nanmax(df[emb_cols], axis=0)

    keep_mask = col_max >= np.float32(threshold)
    cols_to_drop = np.array(emb_cols)[~keep_mask].tolist()

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    del col_max
    gc.collect()

    print(
        f"Removed {len(cols_to_drop)} columns "
        f"(plus {len(dropped_zero) if zero_only_drop else 0} all-zero). "
        f"Final shape: {df.shape}"
    )

if __name__ == '__main__':
    ks = [4, 5, 6, 7]

    for i in tqdm(range(len(ks))):
        kmer_standardize_and_filter(config.DIR_KMER_DATASETS, ks[i], mode=0, seed=1)
    for i in tqdm(range(len(ks))):
        kmer_standardize_and_filter(config.DIR_INCEST_MANY, ks[i])
