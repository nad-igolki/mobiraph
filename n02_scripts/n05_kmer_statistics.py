import config


import numpy as np
import json


def kmer_distribution(sequence: str, k: int):
    """
    Возвращает нормализованный вектор распределения k-меров в последовательности

    Порядок k-меров фиксирован: лексикографический по 'A', 'C', 'G', 'T'

    Parameters
    ----------
    sequence : str
        Последовательность нуклеотидов (A, C, G, T).
    k : int
        Длина k-мера.

    Returns
    -------
    kmers : list[str]
        Список всех возможных k-меров в фиксированном порядке.
    embedding : np.ndarray
        Нормализованный вектор распределения k-меров.
    """
    with open(config.FILE_KMERS, "r", encoding="utf-8") as f:
        kmers_all = json.load(f)
    kmers = kmers_all[k]

    # Словарь для подсчёта
    counts = {kmer: 0 for kmer in kmers}

    # Считаем k-меры в последовательности
    total_kmers = len(sequence) - k + 1
    if total_kmers <= 0:
        # Если последовательность слишком короткая
        return kmers, np.zeros(len(kmers))

    for i in range(total_kmers):
        kmer = sequence[i:i + k]
        if kmer in counts:
            counts[kmer] += 1

    # Преобразуем в вектор в том же порядке, что и kmers
    embedding = np.array([counts[kmer] for kmer in kmers], dtype=float)

    # Нормализация
    embedding /= total_kmers

    return kmers, embedding
