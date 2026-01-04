import config

import os


suffixes = ['_seq_sv_large.fasta', '_sv_large_te_class.txt', '_sv_families.txt']
insects = set()

for filename in os.listdir(config.DIR_GRAPH_PARTITION):
    for suffix in suffixes:
        if filename.endswith(suffix):
            insect_name = filename.rsplit(suffix, 1)[0]
            insects.add(insect_name)


output_path = os.path.join(config.DIR_GRAPH_PARTITION, 'names.txt')
with open(output_path, 'w') as f:
    for insect in sorted(insects):
        f.write(insect + '\n')

print(f'Saved names into file: {output_path}')