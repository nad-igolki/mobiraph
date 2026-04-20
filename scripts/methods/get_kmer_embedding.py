from scripts.n07_kmer_fast_calc import process_k
import os

def get_kmer_embedding(k: int, fasta_path: str, output_dir: str, processes: int | None = None, chunk: int = 64):
    os.makedirs(output_dir, exist_ok=True)
    process_k(k, fasta_path, output_dir, processes, chunk)