import os
from generate_dataset.create_real_test import create_graphs_from_sequences

directory = '/Users/nad/hse/semester06/proj2/src/data_graph_annotated_new'

names_file = os.path.join(directory, 'names.txt')

with open(names_file, 'r') as f:
    insect_names = [line.strip() for line in f if line.strip()]

for name in insect_names:
    print(name)
    fasta_file = os.path.join(directory, f'{name}_target_sequences.fasta')
    if os.path.exists(fasta_file) and os.path.getsize(fasta_file) > 0:
        dir_name = f"{name}_filtered"
        labels_file = os.path.join(directory, f'{name}_target_labels.txt')
        with open(labels_file, 'r') as f:
            labels = [line.strip() for line in f if line.strip()]
        create_graphs_from_sequences(fasta_file, dir_name, labels=labels)