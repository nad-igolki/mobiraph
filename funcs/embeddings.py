import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh, svds
from node2vec import Node2Vec
from funcs.parsing import count_specific_kmers


# 1. Laplacian Eigenmaps
def laplacian_eigenmaps_graph(graph, dim=10):
    # Возвращает dim собственных значений Лапласиана графа.
    laplacian = nx.normalized_laplacian_matrix(graph).astype(float)
    new_dim = dim
    if laplacian.shape[0] <= dim:
        new_dim = laplacian.shape[0] - 1
        # print(f'Reduced to {dim}')
    eigenvalues, _ = eigsh(laplacian, k=new_dim, which='SM')  # Собственные значения
    if len(eigenvalues) < dim:
        eigenvalues = np.pad(eigenvalues, (0, dim - len(eigenvalues)), mode='constant', constant_values=0)
    return eigenvalues


# 2. AROPE
def arope_graph_embedding(graph, dim=10):
    # Возвращает dim сингулярных значений матрицы смежности графа.
    adjacency = nx.to_numpy_array(graph)
    min_size = min(adjacency.shape)
    new_dim = dim
    if min_size <= dim:
        new_dim = min_size - 1
    _, s, _ = svds(adjacency, k=new_dim)  # Сингулярные значения
    if len(s) < dim:
        s = np.pad(s, (0, dim - len(s)), mode='constant', constant_values=0)
    return s


# 3. Node2Vec
def node2vec_graph_embedding(graph, dim=10):
    node2vec = Node2Vec(graph, dimensions=dim, walk_length=80, num_walks=80, workers=10)
    model = node2vec.fit(window=10, min_count=1, batch_words=200)
    # Получение эмбеддингов для узлов
    node_embeddings = {str(node): model.wv[str(node)] for node in graph.nodes if str(node) in model.wv}
    embeddings = np.array(list(node_embeddings.values()))
    mean_embedding = np.mean(embeddings, axis=0)
    return mean_embedding


# 4. Добавление характеристик
def characteristics(graph, node_freq_dict, random_kmers, node2name_dict, name2seq):
    degrees = [deg for _, deg in graph.degree()]
    mean_degree = sum(degrees) / len(degrees) if len(degrees) > 0 else 0

    number_nodes = len(graph.nodes())

    max_degrees = sorted(degrees)[:3]
    max_degrees = list(np.pad(max_degrees, (0, 3 - len(max_degrees)), mode='constant', constant_values=0))

    freqs = sorted([node_freq_dict[node] for node in list(graph.nodes())], reverse=True)[:3]
    freqs = list(np.pad(freqs, (0, 3 - len(freqs)), mode='constant', constant_values=0))

    kmer_counts = [0] * len(random_kmers)

    for node in list(graph.nodes()):
        name = node2name_dict[node]
        sequence = name2seq[name]
        kmer_count = count_specific_kmers(sequence, random_kmers)

        kmer_counts = [x + y for x, y in zip(kmer_counts, kmer_count)]

    sum_len = 0
    for node in list(graph.nodes()):
        name = node2name_dict[node]
        sequence = name2seq[name]
        len_seq = len(sequence)
        sum_len += len_seq

    kmer_counts = [x / sum_len for x in kmer_counts]

    min_degree_count = 0
    for _, deg in graph.degree():
        if deg == 1:
            min_degree_count += 1

    embedding = [mean_degree, number_nodes, min_degree_count] + max_degrees + freqs + kmer_counts
    return embedding


# Характеристики для каждой вершины
def node_characteristics(node, graph, node_freq_dict, random_kmers, node2name_dict, name2seq):
    degree = graph.degree(node)
    freq = node_freq_dict[node]

    kmer_counts = [0] * len(random_kmers)

    name = node2name_dict[node]
    sequence = name2seq[name]
    kmer_count = count_specific_kmers(sequence, random_kmers)
    kmer_counts = [x + y for x, y in zip(kmer_counts, kmer_count)]
    name = node2name_dict[node]
    sequence = name2seq[name]
    len_seq = len(sequence)
    kmer_counts = [x / len_seq for x in kmer_counts]


    neighbors = list(graph.neighbors(node))
    total_degree = sum(graph.degree(neighbor) for neighbor in neighbors) / len(neighbors)



    embedding = [degree, freq, total_degree] + kmer_counts
    return embedding