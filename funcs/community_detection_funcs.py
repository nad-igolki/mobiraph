import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import leidenalg
import igraph as ig
from collections import defaultdict

def community_detection(G):
    G_igraph = ig.Graph.from_networkx(G)
    partition = leidenalg.find_partition(G_igraph, leidenalg.ModularityVertexPartition)
    clusters = partition.membership
    for node, cluster in zip(G.nodes(), clusters):
        G.nodes[node]['cluster'] = cluster

    clusters_dict = defaultdict(set)
    for node, cluster in zip(G.nodes(), clusters):
        clusters_dict[cluster].add(node)

    return clusters_dict


# Рекурсивный partition, чтобы получить классы поменьше
def community_detection_recursive_with_limit(G, max_size=10, max_depth=10):
    def detect_communities(graph):
        # Кластеризация с помощью leiden
        graph_igraph = ig.Graph.from_networkx(graph)
        partition = leidenalg.find_partition(graph_igraph, leidenalg.ModularityVertexPartition)
        clusters = partition.membership
        # Сгруппировываем узлы по кластерам
        clusters_dict = defaultdict(set)
        for node, cluster in zip(graph.nodes(), clusters):
            clusters_dict[cluster].add(node)
        return clusters_dict

    def split_large_clusters(graph, clusters_dict, current_depth):
        if current_depth > max_depth:
            return clusters_dict  # Возвращаем текущие кластеры, если глубина достигнута

        new_clusters_dict = {}
        cluster_id = 0

        for cluster_nodes in clusters_dict.values():
            if len(cluster_nodes) <= max_size:
                new_clusters_dict[cluster_id] = cluster_nodes
                cluster_id += 1
            else:
                # Создать подграф для большого кластера
                subgraph = graph.subgraph(cluster_nodes)
                # Кластеризация для подграфа
                sub_clusters_dict = detect_communities(subgraph)
                sub_clusters = split_large_clusters(subgraph, sub_clusters_dict, current_depth + 1)
                for nodes in sub_clusters.values():
                    new_clusters_dict[cluster_id] = nodes
                    cluster_id += 1

        return new_clusters_dict

    initial_clusters = detect_communities(G)
    # Рекурсивное разбиение
    final_clusters = split_large_clusters(G, initial_clusters, current_depth=0)
    for cluster_id, nodes in final_clusters.items():
        for node in nodes:
            G.nodes[node]['cluster'] = cluster_id

    return final_clusters


