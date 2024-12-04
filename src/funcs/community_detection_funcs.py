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