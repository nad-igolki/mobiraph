import numpy as np

mapping = {"A": 0, "C": 1, "G": 2, "T": 3}

def one_hot(seq):
    arr = np.zeros((40, 4), dtype=np.uint8)
    for i, nt in enumerate(seq[:40].upper()):
        if nt in mapping:
            arr[i, mapping[nt]] = 1
    return arr

def process_fasta(in_fasta, out_npz):
    names = []
    embeddings = []

    name = None
    seq = ""

    def save_current():
        if name is not None:
            s = seq[:20] + seq[-20:]
            names.append(name)
            embeddings.append(one_hot(s))

    with open(in_fasta) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                save_current()
                name = line[1:].split()[0]
                seq = ""
            else:
                seq += line

        save_current()

    np.savez(
        out_npz,
        names=np.array(names),
        embeddings=np.array(embeddings)
    )
process_fasta("/Users/nad/mobiraph/data/insect_sv_fam_best.fasta", "/Users/nad/mobiraph/data/n26_sv_processed/20_symbols_insects.npz")