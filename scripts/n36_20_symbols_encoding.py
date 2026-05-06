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



input_file = "/Users/nad/mobiraph/data/n13_repbase_processed/all_sequences_filtered_02_ltr_correction.fasta"
output_file = "/Users/nad/mobiraph/data/n13_repbase_processed/all_sequences_30.fasta"

with open(input_file) as f, open(output_file, "w") as out:
    seq_id = None
    seq = []

    for line in f:
        line = line.strip()
        if line.startswith(">"):
            if seq_id is not None:
                sequence = "".join(seq)

                start = sequence[:35]
                end = sequence[-35:]

                out.write(f">{seq_id}_start\n{start}\n")
                out.write(f">{seq_id}_end\n{end}\n")

            seq_id = line[1:].split()[0]
            seq = []
        else:
            seq.append(line)

    if seq_id is not None:
        sequence = "".join(seq)

        start = sequence[:30]
        end = sequence[-30:]

        out.write(f">{seq_id}_start\n{start}\n")
        out.write(f">{seq_id}_end\n{end}\n")



import pandas as pd

input_csv = "/Users/nad/mobiraph/data/n12_all_sequences_kmer/7_30.csv"
output_csv = "/Users/nad/mobiraph/data/n12_all_sequences_kmer/7_30_merged.csv"

df = pd.read_csv(input_csv)

df["seq_id"] = df["name"].str.replace(r"_(start|end)$", "", regex=True)
df["part"] = df["name"].str.extract(r"_(start|end)$")

embedding_cols = [c for c in df.columns if c.startswith("emb_")]

start_df = df[df["part"] == "start"][["seq_id"] + embedding_cols].copy()
end_df = df[df["part"] == "end"][["seq_id"] + embedding_cols].copy()

start_df = start_df.rename(columns={c: c for c in embedding_cols})

end_df = end_df.rename(
    columns={
        c: f"emb_{int(c.split('_')[1]) + 256}"
        for c in embedding_cols
    }
)

merged = pd.merge(start_df, end_df, on="seq_id", how="inner")
merged.rename(columns={'seq_id': 'name'}, inplace=True)

merged.to_csv(output_csv, index=False)


from functools import reduce
import pandas as pd

df1 = pd.read_csv("/Users/nad/mobiraph/data/n12_all_sequences_kmer/1.csv")
df2 = pd.read_csv("/Users/nad/mobiraph/data/n12_all_sequences_kmer/2.csv")
df3 = pd.read_csv("/Users/nad/mobiraph/data/n12_all_sequences_kmer/3.csv")

dfs = [df1, df2, df3]
df_merged = reduce(lambda left, right: left.merge(right, on="name"), dfs)

df_merged.to_csv("/Users/nad/mobiraph/data/n12_all_sequences_kmer/1_2_3.csv", index=False)



mapping = {
    'BEL': 'Bel-Pao',
    'CR1': 'CR1',
    'Copia': 'Copia',
    'DIRS': 'DIRS',
    'EnSpm/CACTA': 'CACTA',
    'Gypsy': 'Gypsy',
    'Harbinger': 'PIF-Harbinger',
    'Helitron': 'Helitron',
    'L1': 'L1',
    'Mariner/Tc1': 'Tc1-Mariner',
    'MuDR': 'Mutator',
    'RTE': 'RTE',
    'RTEX': 'RTEX',
    'SINE': 'tRNA',
    'Tad1': 'Tad1',
    'Tx1': 'Tx1',
    'hAT': 'hAT'
}

import json
with open('/Users/nad/mobiraph/data/n13_repbase_processed/metadata_03.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

names = []
with open('/Users/nad/mobiraph/data/n13_repbase_processed_wo_bad_sf/id_train_with_superfamilies.txt', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        names.append(line[:-1])

import csv
with open("train_labels.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    # при необходимости — заголовки
    writer.writerow(["column1", "column2"])

    # построчная запись
    for name in names:
        writer.writerow([name, mapping[metadata[name]['superfamily']]])



all_sup = [metadata[name]['superfamily'] for name in names]






import csv

csv_file = "train_labels.csv"
fasta_file = "/Users/nad/mobiraph/data/n13_repbase_processed/all_sequences_filtered_02_ltr_correction.fasta"
output_fasta = "/Users/nad/mobiraph/data/n13_repbase_processed/all_sequences.ref"

# Названия колонок в CSV
HEADER_COL = "column1"
CLASS_COL = "column2"


def read_csv_pairs(csv_path):
    """
    Возвращает словарь:
    {заголовок_из_csv: класс}
    """
    pairs = {}

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            header = row[HEADER_COL].strip()
            cls = row[CLASS_COL].strip()
            pairs[header] = cls

    return pairs


def parse_fasta(fasta_path):
    """
    Построчно читает FASTA и возвращает пары:
    заголовок, последовательность
    """
    header = None
    seq_parts = []

    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)

                header = line[1:]  # убираем >
                seq_parts = []
            else:
                seq_parts.append(line)

        if header is not None:
            yield header, "".join(seq_parts)


import json

with open("/Users/nad/mobiraph/data/n13_repbase_processed/metadata_02_ltr_correction.json", "r", encoding="utf-8") as f:
    dict = json.load(f)

print(dict)


def write_filtered_fasta(csv_path, fasta_path, output_path):
    csv_pairs = read_csv_pairs(csv_path)

    with open(output_path, "w", encoding="utf-8") as out:
        for fasta_header, sequence in parse_fasta(fasta_path):

            # если в FASTA заголовок полностью совпадает с заголовком из CSV
            if fasta_header in csv_pairs:
                cls = csv_pairs[fasta_header]

                new_header = f"{fasta_header}\t{cls}\t{dict[fasta_header]['organism']}"

                out.write(f">{new_header}\n")
                out.write(f"{sequence}\n")


write_filtered_fasta(csv_file, fasta_file, output_fasta)






from Bio import SeqIO

input_fasta = "/Users/nad/mobiraph/data/n13_repbase_processed/all_sequences_ltr_correction_test.fasta"
output_fasta = "/Users/nad/mobiraph/data/n13_repbase_processed/all_sequences_ltr_correction_test_unified.fasta"
mapping_file = "/Users/nad/mobiraph/data/n13_repbase_processed/all_sequences_ltr_correction_test_mapping.csv"

records = []
mapping = []

for i, record in enumerate(SeqIO.parse(input_fasta, "fasta"), start=1):
    old_name = record.id
    new_name = f"name{i}"

    mapping.append((new_name, old_name))

    record.id = new_name
    record.name = new_name
    record.description = new_name

    records.append(record)

SeqIO.write(records, output_fasta, "fasta")

with open(mapping_file, "w") as f:
    f.write("new_name\told_name\n")
    for new, old in mapping:
        f.write(f"{new}\t{old}\n")




def fasta_to_txt(input_fasta, output_txt):
    with open(input_fasta, 'r') as fasta, open(output_txt, 'w') as txt:
        header = ""
        sequence = []

        for line in fasta:
            line = line.strip()

            if line.startswith(">"):
                # сохранить предыдущую запись
                if header:
                    txt.write(f"{header},{''.join(sequence)}\n")

                # убрать >
                full_header = line[1:]

                # взять второе значение после split по tab
                parts = full_header.split("\t")
                header = parts[1] if len(parts) > 1 else parts[0]

                sequence = []
            else:
                sequence.append(line)

        # последняя запись
        if header:
            txt.write(f"{header},{''.join(sequence)}\n")


# пример
fasta_to_txt("/Users/nad/mobiraph/data/n13_repbase_processed/all_sequences.ref", "/Users/nad/mobiraph/data/n13_repbase_processed/ipt_shuffle_Custom_CNN_data.txt")