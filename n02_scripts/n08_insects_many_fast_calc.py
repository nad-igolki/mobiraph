import os
import csv
from tqdm import tqdm
import multiprocessing as mp
import config
from n02_scripts.n05_kmer_statistics import kmer_distribution

FASTA_PATH = config.FILE_INSECT_MANY_FASTA
OUTPUT_PATH = config.DIR_INCEST_MANY

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
                name = line[1:].strip()
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
    # ks = [4, 5, 6, 7]
    ks = [5, 6, 7]
    for k in ks:
        process_k(k, FASTA_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
