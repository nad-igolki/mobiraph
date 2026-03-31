import os
import csv
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import json
from collections import Counter

FASTA_PATH = ''
OUTPUT_PATH = ''

FILE_KMERS = ''

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
    with open(FILE_KMERS, "r", encoding="utf-8") as f:
        kmers_all = json.load(f)
    kmers = kmers_all[str(k)]

    # Считаем k-меры в последовательности
    total_kmers = len(sequence) - k + 1
    if total_kmers <= 0:
        # Если последовательность слишком короткая
        return kmers, np.zeros(len(kmers))

    counts = Counter(sequence[i:i+k] for i in range(total_kmers))

    # Преобразуем в вектор в том же порядке, что и kmers
    embedding = np.array([counts[kmer] for kmer in kmers], dtype=float)

    # Нормализация
    embedding /= total_kmers

    return kmers, embedding


def read_fasta(path: str):
    name = None
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name = line[1:].split("\t")[0]
                chunks = []
            else:
                chunks.append(line.upper())
        if name is not None:
            yield name, "".join(chunks)

def _embed_one(args):
    name, seq, k = args
    _, emb = kmer_distribution(seq, k)
    return name, emb

def process_k(k: int, fasta_path: str, out_dir: str, processes: int | None = None, chunk: int = 64):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{k}.csv")
    fasta_iter = read_fasta(fasta_path)
    try:
        first_name, first_seq = next(fasta_iter)
    except StopIteration:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            pass
        return

    _, emb0 = kmer_distribution(first_seq, k)
    emb_len = len(emb0)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["name"] + [f"emb_{i}" for i in range(emb_len)]
        writer.writerow(header)
        writer.writerow([first_name] + list(emb0))

        if processes is None:
            processes = max(1, mp.cpu_count() - 1)

        with mp.get_context("spawn").Pool(processes) as pool:

            args_iter = ((name, seq, k) for (name, seq) in fasta_iter)
            for name, emb in tqdm(pool.imap_unordered(_embed_one, args_iter, chunksize=chunk), desc=f"k={k}"):
                writer.writerow([name] + list(emb))


def main():
    ks = [4, 5, 6, 7]
    for k in ks:
        process_k(k, FASTA_PATH, OUTPUT_PATH, processes=30)


if __name__ == "__main__":
    main()
