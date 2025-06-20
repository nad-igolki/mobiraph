import os
import csv


src_dir = "test_LTR"
dst_train = "real_LTR_train"
dst_test = "real_LTR_test"
n_transfer = 400

os.makedirs(dst_train, exist_ok=True)
os.makedirs(dst_test, exist_ok=True)

def split_graph_ids(labels_file, n):
    graph_ids = []
    with open(labels_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            graph_ids.append(int(row['graph_id']))
    graph_ids = sorted(set(graph_ids))  # отсортированные
    return set(graph_ids[:n]), set(graph_ids[n:])

train_ids, test_ids = split_graph_ids(os.path.join(src_dir, 'graph_labels.csv'), n_transfer)

def split_file(file_name, id_field):
    src_path = os.path.join(src_dir, file_name)
    train_path = os.path.join(dst_train, file_name)
    test_path = os.path.join(dst_test, file_name)

    with open(src_path, 'r') as src_file, \
         open(train_path, 'w', newline='') as train_file, \
         open(test_path, 'w', newline='') as test_file:

        reader = csv.DictReader(src_file)
        train_writer = csv.DictWriter(train_file, fieldnames=reader.fieldnames)
        test_writer = csv.DictWriter(test_file, fieldnames=reader.fieldnames)

        train_writer.writeheader()
        test_writer.writeheader()

        for row in reader:
            gid = int(row[id_field])
            if gid in train_ids:
                train_writer.writerow(row)
            elif gid in test_ids:
                test_writer.writerow(row)


def split_fasta_sequences(fasta_path, train_path, test_path, train_ids):
    with open(fasta_path, 'r') as fasta_file, \
         open(train_path, 'w') as train_file, \
         open(test_path, 'w') as test_file:

        current_id = None
        current_lines = []

        for line in fasta_file:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    output = train_file if current_id in train_ids else test_file
                    output.write('\n'.join(current_lines) + '\n')

                current_id = int(line[1:].strip())
                current_lines = [line]  # начинаем новую последовательность
            else:
                current_lines.append(line)

        # сохраняем последнюю последовательность
        if current_id is not None:
            output = train_file if current_id in train_ids else test_file
            output.write('\n'.join(current_lines) + '\n')


split_file('graph_labels.csv', 'graph_id')
split_file('nodes.csv', 'graph_id')
split_file('edges.csv', 'graph_id')

fasta_input = os.path.join(src_dir, 'sequences.csv')
fasta_train = os.path.join(dst_train, 'sequences.csv')
fasta_test = os.path.join(dst_test, 'sequences.csv')
split_fasta_sequences(fasta_input, fasta_train, fasta_test, train_ids)


dir1 = "LTR_NO"
dir2 = "real_LTR_train"
dir_combined = 'LTR_NO_with_real'
os.makedirs(dir_combined, exist_ok=True)

def get_sorted_graph_ids(path):
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        return sorted(int(row['graph_id']) for row in reader)


def get_max_graph_id(path):
    return max(get_sorted_graph_ids(path))


def get_graph_ids_to_exclude(n):
    ids = get_sorted_graph_ids(os.path.join(dir1, 'graph_labels.csv'))
    return set(str(i) for i in ids[:n])

def get_max_graph_id(labels_path):
    with open(labels_path, 'r') as f:
        reader = csv.DictReader(f)
        return max(int(row['graph_id']) for row in reader)

max_graph_id = get_max_graph_id(os.path.join(dir1, 'graph_labels.csv'))
exclude_ids = get_graph_ids_to_exclude(n_transfer)
offset = max_graph_id + 1

# объединение в csv с корректировкой graph_id
def combine_csv(file_name, id_fields):
    output_path = os.path.join(dir_combined, file_name)
    writer = None

    with open(output_path, 'w', newline='') as out_f:
        for src_dir in [dir1, dir2]:
            file_path = os.path.join(src_dir, file_name)
            if not os.path.exists(file_path):
                continue

            with open(file_path, 'r') as in_f:
                reader = csv.DictReader(in_f)
                if writer is None:
                    writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames)
                    writer.writeheader()

                for row in reader:
                    if src_dir == dir1 and row['graph_id'] in exclude_ids:
                        continue  # исключаем первые n_transfer графов
                    if src_dir == dir2:
                        for field in id_fields:
                            if row[field]:
                                row[field] = str(int(row[field]) + offset)
                    writer.writerow(row)



def combine_sequences_fasta(file_name):
    output_path = os.path.join(dir_combined, file_name)
    input_paths = [os.path.join(dir1, file_name), os.path.join(dir2, file_name)]

    removed_ids = set(map(int, exclude_ids))

    with open(output_path, 'w') as out_f:
        for i, input_path in enumerate(input_paths):
            if not os.path.exists(input_path):
                continue

            with open(input_path, 'r') as in_f:
                current_id = None
                current_lines = []

                for line in in_f:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_lines:
                            out_f.write('\n'.join(current_lines) + '\n')
                        current_id = int(line[1:])
                        if i == 0 and current_id in removed_ids:
                            current_id = None
                            current_lines = []
                            continue
                        if i == 1:
                            current_id += offset
                        current_lines = [f'>{current_id}']
                    else:
                        if current_id is not None:
                            current_lines.append(line)

                if current_lines:
                    out_f.write('\n'.join(current_lines) + '\n')


combine_csv('graph_labels.csv', ['graph_id'])
combine_csv('nodes.csv', ['graph_id'])
combine_csv('edges.csv', ['graph_id'])
combine_sequences_fasta('sequences.csv')