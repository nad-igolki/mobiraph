import numpy as np
from generate_dataset.generate import generate_LTR, generate_TIR, generate_NO, generate_INNER
from tqdm import tqdm
import csv

import os
import pandas as pd


def nucleotide_fractions(seq):
    total = len(seq)
    return [
        np.count_nonzero(seq == 'A') / total,
        np.count_nonzero(seq == 'T') / total,
        np.count_nonzero(seq == 'G') / total,
        np.count_nonzero(seq == 'C') / total,
    ]


def identical_fractions(seq):
    return [
        0.25, 0.25, 0.25, 0.25
    ]


def create_edges_csv_start(filename='generated_graphs/edges.csv'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["graph_id", "src", "dst", "edge_param"])

def append_edges_to_csv(graph_id, adj, node_index_dic, filename='generated_graphs/edges.csv', wsize=15):
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        src, dst = np.nonzero(adj)
        for u, v in zip(src, dst):
            if adj[u].sum() == adj[0][0] or adj[v].sum() == adj[0][0]:
                continue
            writer.writerow([graph_id, node_index_dic[u], node_index_dic[v], adj[u][v] / wsize])


def create_nodes_csv_start(filename='generated_graphs/nodes.csv'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        columns = ['graph_id', 'node_id'] + [f'feat_{i}' for i in range(5)]
        writer.writerow(columns)


def append_nodes_to_csv(graph_id, features, adj, filename='generated_graphs/nodes.csv'):
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        num_nodes = features.shape[0]

        node_index = 0
        node_index_dic = {}

        for node_id in range(num_nodes):
            node_index_dic[node_id] = node_index
            if adj[node_id].sum() == adj[0][0]:
                continue
            feature = identical_fractions(features[node_index]) + [node_id / num_nodes]
            row = [graph_id, node_index] + feature
            writer.writerow(row)
            node_index += 1
    return node_index_dic


def create_graph_labels_csv(labels, filename='generated_graphs/graph_labels.csv'):
    df = pd.DataFrame({
        "graph_id": list(range(len(labels))),
        "label": labels
    })
    df.to_csv(filename, index=False)
    print(f"файл успешно сохранён: {filename}")


def append_to_fasta(seq_id, sequence, fasta_path):
    if not os.path.exists(fasta_path):
        open(fasta_path, 'w').close()

    header = f">{seq_id}"

    with open(fasta_path, 'a') as f:
        f.write(f"{header}\n")
        f.write(f"{sequence}\n")
