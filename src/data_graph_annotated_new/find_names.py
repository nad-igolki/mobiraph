import os

directory = '/Users/nad/hse/semester06/proj2/src/data_graph_annotated_new'

suffixes = ['_sv_te_class', '_node_partition', '_seq.fasta']
insects = set()

for filename in os.listdir(directory):
    for suffix in suffixes:
        if filename.endswith(suffix):
            insect_name = filename.rsplit(suffix, 1)[0]
            insects.add(insect_name)

print(sorted(insects))


output_path = os.path.join(directory, 'names.txt')
with open(output_path, 'w') as f:
    for insect in sorted(insects):
        f.write(insect + '\n')

print(f'Список сохранён в файл: {output_path}')