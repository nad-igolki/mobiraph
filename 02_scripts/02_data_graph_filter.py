import config

import os
import pandas as pd


names_file = os.path.join(config.DIR_GRAPH_PARTITION, 'names.txt')

with open(names_file, 'r') as f:
    insect_names = [line.strip() for line in f if line.strip()]

name_to_class = {}
name_to_partition = {}
name_to_sequence = {}

for name in insect_names:
    class_file = os.path.join(config.DIR_GRAPH_PARTITION, f'{name}_sv_te_class.txt')
    partition_file = os.path.join(config.DIR_GRAPH_PARTITION, f'{name}_node_partition.txt')
    fasta_file = os.path.join(config.DIR_GRAPH_PARTITION, f'{name}_seq.fasta')

    if os.path.exists(class_file):
        name_to_class[name] = pd.read_csv(class_file, sep='\t')
    else:
        name_to_class[name] = None

    if os.path.exists(partition_file):
        partition_dict = {}
        with open(partition_file, 'r') as f:
            for line in f:
                if '\t' in line:
                    key, value = line.strip().split('\t', 1)
                    partition_dict[key] = int(value)
        name_to_partition[name] = partition_dict
    else:
        name_to_partition[name] = None

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
        name_to_sequence[name] = sequences
    else:
        name_to_sequence[name] = None


    components_to_take = []
    names_class_to_take = []
    if name_to_class[name] is not None and 'sv' in name_to_class[name]:
        for node_name in name_to_class[name]['sv']:
            if name_to_partition[name][node_name] not in components_to_take:
                components_to_take.append(name_to_partition[name][node_name])
                type_value = name_to_class[name].loc[name_to_class[name]['sv'] == node_name, 'type'].values[0]
                names_class_to_take.append([node_name, type_value])

        output_fasta = os.path.join(config.DIR_SEQUENCES, f'{name}_sequences.fasta')
        with open(output_fasta, 'w') as f:
            for [node_name, type_value] in names_class_to_take:
                f.write(f'>{node_name}_{name_to_partition[name][node_name]}\n')
                seq = name_to_sequence[name][node_name]
                for i in range(0, len(seq), 60):
                    f.write(seq[i:i + 60] + '\n')

        output_labels = os.path.join(config.DIR_SEQUENCES, f'{name}_labels.txt')
        with open(output_labels, 'w') as f:
            for [node_name, type_value] in names_class_to_take:
                f.write(type_value + '\n')

        print(f'Saved into files: {output_fasta}, {output_labels}')

