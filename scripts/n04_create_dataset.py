import numpy as np
import pandas as pd

def create_dataset(df: pd.DataFrame, mode: int, seed: int | None = None) -> pd.DataFrame:
    """
    Собирает датасет из исходного DataFrame, выбирая по ОДНОЙ строке
    для каждого уникального partition_id.

    Параметры
    ---------
    df : pd.DataFrame
    mode : int
        0 — случайный выбор внутри каждого partition_id (использует seed);
        1 — максимальная длина (max по 'length');
        2 — минимальная длина (min по 'length');
        3 — медианная длина (строка с длиной, ближайшей к median по группе).
    seed : int | None
        Используется только при mode=0 для воспроизводимости.

    Возвращает
    ----------
    pd.DataFrame
        Ровно по одной строке на каждый уникальный partition_id.
    """

    if mode not in (0, 1, 2, 3):
        raise ValueError("mode должен быть одним из {0, 1, 2, 3}")
    if mode == 0 and seed is None:
        raise ValueError("Для mode=0 необходимо указать seed (целое число).")

    selected_idx = []
    rng = np.random.default_rng(seed) if mode == 0 else None

    # Чтобы порядок и воспроизводимость были стабильными
    # обходим partition_id в отсортированном порядке.
    for pid, g in df.sort_values("partition_id").groupby("partition_id", sort=True):
        if mode == 0:
            # Случайный выбор 1 строки в группе с детерминированным seed
            pos = rng.integers(len(g))
            selected_idx.append(g.index[pos])

        elif mode == 1:
            # Макс длина
            selected_idx.append(g["length"].idxmax())

        elif mode == 2:
            # Мин длина
            selected_idx.append(g["length"].idxmin())

        else:  # mode == 3
            # Ближайшая к медиане длина; при равенстве — первая по индексу
            med = g["length"].median()
            idx = (g["length"] - med).abs().idxmin()
            selected_idx.append(idx)

    result = df.loc[selected_idx].reset_index(drop=True)
    return result
