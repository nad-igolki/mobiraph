from generate_dataset.write_csv_funcs import *
from utils.fast_dot_plot_dna import dotplot


wsize = 15
nmatch = 12
scatter = False


def read_fasta_generator(filename):
    with open(filename, "r") as f:
        seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    yield "".join(seq)
                    seq = []
            else:
                seq.append(line)
        if seq:
            yield "".join(seq)


def create_graphs_from_sequences(seq_file_name, dir_name, labels=None):
    if labels is None:
        labels = []
    file_edges = f'generated_graphs/{dir_name}/edges.csv'
    file_nodes = f'generated_graphs/{dir_name}/nodes.csv'
    file_labels = f'generated_graphs/{dir_name}/graph_labels.csv'
    file_for_sequences = f'generated_graphs/{dir_name}/sequences.csv'

    create_edges_csv_start(filename=file_edges)
    create_nodes_csv_start(filename=file_nodes)

    max_len = 10000
    graph_id = 0

    for seq in tqdm(read_fasta_generator(seq_file_name), desc="Number of sequences"):
        if len(seq) > max_len:
            continue
        features, matrix = dotplot(seq, seq, wsize=wsize, nmatch=nmatch, scatter=False)
        node_index_dic = append_nodes_to_csv(graph_id, features, matrix, filename=file_nodes)
        append_edges_to_csv(graph_id, matrix, node_index_dic, wsize=wsize, filename=file_edges)
        append_to_fasta(graph_id, seq, file_for_sequences)
        graph_id += 1
        if labels is None:
            labels.append(0)
    create_graph_labels_csv(labels, filename=file_labels)



# ltr_file = "test_data/arabidopsis_LTR.fasta"
# line_file = "test_data/arabidopsis_LINE.fasta"
# dir_name = "test_LTR_LINE"
# create_graphs_from_sequences(ltr_file, dir_name)
# create_graphs_from_sequences(line_file, dir_name)