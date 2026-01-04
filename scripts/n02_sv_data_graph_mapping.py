import config

import os
import pandas as pd
import re
import csv
from tqdm import tqdm


names_file = os.path.join(config.DIR_GRAPH_PARTITION, 'names.txt')

with open(names_file, 'r') as f:
    insect_names = [line.strip() for line in f if line.strip()]

name_to_class = {}
name_to_partition = {}
name_to_sequence = {}

header = ['name', 'insect', 'partition', 'type', 'length']
rows = []

for insect_name in tqdm(insect_names, desc="Processing insects"):
    class_file = os.path.join(config.DIR_GRAPH_PARTITION, f'{insect_name}_sv_large_te_class.txt')
    partition_file = os.path.join(config.DIR_GRAPH_PARTITION, f'{insect_name}_sv_families.txt')
    fasta_file = os.path.join(config.DIR_GRAPH_PARTITION, f'{insect_name}_seq_sv_large.fasta')

    if os.path.exists(class_file):
        name_to_class[insect_name] = pd.read_csv(class_file, sep='\t')
    else:
        name_to_class[insect_name] = None

    if os.path.exists(partition_file):
        partition_dict = {}
        with open(partition_file, 'r') as f:
            for line in f:
                if '\t' in line:
                    key, value = line.strip().split('\t', 1)
                    partition_dict[key] = int(value)
        name_to_partition[insect_name] = partition_dict
    else:
        name_to_partition[insect_name] = None

    if os.path.exists(fasta_file):
        sequences = {}
        with open(fasta_file, 'r') as f:
            current_header = None
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_header is not None:
                        sequences[current_header] = ''.join(current_seq)
                    current_header = line[1:]  # убираем >
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_header is not None:
                sequences[current_header] = ''.join(current_seq)
        name_to_sequence[insect_name] = sequences
    else:
        name_to_sequence[insect_name] = None


    for name_value in name_to_sequence[insect_name].keys():
        insect_value = insect_name
        partition_value = (
            int(name_to_partition[insect_name][name_value])
            if insect_name in name_to_partition.keys() and name_to_partition[insect_name] is not None and name_value in name_to_partition[insect_name].keys()
            else None
        )
        try:
            # print(name_to_class[insect_name])
            class_df = name_to_class[insect_name]
            if class_df is not None:
                # print(class_df['SV'], name_value)
                filtered_class_df = class_df.loc[class_df['SV'] == name_value, 'MainType']
                type_value = filtered_class_df.values[0]
            else:
                type_value = None
        except (KeyError, IndexError):
            type_value = None
        length_value = re.search(r'\|(\d+)$', name_value).group(1)
        rows.append([name_value, insect_value, partition_value, type_value, length_value])


with open(config.FILE_INSECT_MAP, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)