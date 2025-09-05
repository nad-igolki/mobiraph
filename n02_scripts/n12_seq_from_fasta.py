from Bio import SeqIO
from tqdm import tqdm
import os

def seq_from_fasta(fasta_path: str, seq_names: list, prefer_exact: bool = True):
    found = {}

    all_records = list(SeqIO.parse(fasta_path, "fasta"))

    for name in tqdm(seq_names, desc=f"Extracting sequences from {os.path.basename(fasta_path)}"):
        seq = None

        if prefer_exact:
            for rec in all_records:
                if rec.description == name:
                    seq = str(rec.seq)
                    break

        if seq is None:
            for rec in all_records:
                if rec.description.startswith(name):
                    seq = str(rec.seq)
                    break

        if seq is not None:
            found[name] = seq

    return found
