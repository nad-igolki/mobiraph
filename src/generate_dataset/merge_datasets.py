import os
import csv

def merge_graph_datasets(input_dirs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    edges_out = open(os.path.join(output_dir, 'edges.csv'), 'w', newline='')
    nodes_out = open(os.path.join(output_dir, 'nodes.csv'), 'w', newline='')
    labels_out = open(os.path.join(output_dir, 'graph_labels.csv'), 'w', newline='')
    sequences_out = open(os.path.join(output_dir, 'sequences.fasta'), 'w')

    edges_writer = csv.writer(edges_out)
    nodes_writer = csv.writer(nodes_out)
    labels_writer = csv.writer(labels_out)

    # Пишем заголовки
    edges_writer.writerow(['graph_id', 'src', 'dst', 'edge_param'])
    nodes_writer.writerow(['graph_id', 'node_id', 'feat_0', 'feat_1', 'feat_2', 'feat_3', 'feat_4'])
    labels_writer.writerow(['graph_id', 'label'])

    current_graph_id_offset = 0
    current_label_offset = 0

    for dir_idx, dir_path in enumerate(input_dirs):
        graph_id_map = {}
        label_map = {}

        # Обработка graph_labels.csv
        with open(os.path.join(dir_path, 'graph_labels.csv'), 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                old_gid = int(row['graph_id'])
                old_label = int(row['label'])

                new_gid = old_gid + current_graph_id_offset
                if old_label not in label_map:
                    label_map[old_label] = current_label_offset + len(label_map)

                new_label = label_map[old_label]
                labels_writer.writerow([new_gid, new_label])
                graph_id_map[old_gid] = new_gid

        # Обработка nodes.csv
        with open(os.path.join(dir_path, 'nodes.csv'), 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                old_gid = int(row['graph_id'])
                new_gid = graph_id_map[old_gid]
                nodes_writer.writerow([
                    new_gid,
                    row['node_id'],
                    row['feat_0'], row['feat_1'], row['feat_2'],
                    row['feat_3'], row['feat_4']
                ])

        # Обработка edges.csv
        with open(os.path.join(dir_path, 'edges.csv'), 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                old_gid = int(row['graph_id'])
                new_gid = graph_id_map[old_gid]
                edges_writer.writerow([
                    new_gid,
                    row['src'], row['dst'], row['edge_param']
                ])

        # Обработка sequences.fasta
        seq_path = os.path.join(dir_path, 'sequences.fasta')
        if os.path.exists(seq_path):
            with open(seq_path, 'r') as f:
                current_old_id = None
                buffer = []
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_old_id is not None:
                            new_id = graph_id_map[current_old_id]
                            sequences_out.write(f'>{new_id}\n')
                            sequences_out.write('\n'.join(buffer) + '\n')
                        current_old_id = int(line[1:])
                        buffer = []
                    else:
                        buffer.append(line)

                # Записать последнюю запись
                if current_old_id is not None:
                    new_id = graph_id_map[current_old_id]
                    sequences_out.write(f'>{new_id}\n')
                    sequences_out.write('\n'.join(buffer) + '\n')

        current_graph_id_offset += len(graph_id_map)
        current_label_offset += len(label_map)

    # Закрываем файлы
    edges_out.close()
    nodes_out.close()
    labels_out.close()
    sequences_out.close()


input_dirs = ["LTR_20000", "NO_20000"]
output_dir = "LTR_TIR_20000"
merge_graph_datasets(input_dirs, output_dir)