import config


import itertools
import json


def create_kmer_lists(start=2, end=10):
    nucleotides = ["A", "T", "G", "C"]

    all_sequences = {}

    for k in range(start, end + 1):
        sequences = ["".join(p) for p in itertools.product(nucleotides, repeat=k)]
        all_sequences[k] = sequences

    with open(config.FILE_KMERS, "w", encoding="utf-8") as f:
        json.dump(all_sequences, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    create_kmer_lists()