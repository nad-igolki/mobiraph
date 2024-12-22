import networkx as nx
from collections import Counter
import pandas as pd
import itertools
import random
from collections import Counter

# Предсказание для маленьких компонент brassica juncea

# Загружаем модель
import pickle
file_path = '../../random_forest01.pkl'

print("Чтение...")
with open(file_path, "rb") as f:
    clf = pickle.load(f)
print("Завершено")

# Загружаем граф
filename = '../../data_juncia/graph_collapse_b_juncea.txt'

G = nx.Graph()
G_with_orient = nx.DiGraph()

with open(filename, 'r') as file:
    for line in file:
        node1, node2 = line.strip().split()
        G.add_edge(node1, node2)
        G_with_orient.add_edge(node1, node2)


df = pd.read_csv('../../data_juncia/graph_collapse_node_traits_b_juncea.txt', delimiter='\t')


df['fam'] = df['fam'].replace(['LTR/Copia', 'LTR/Gypsy'], 'LTR')

node_family_dict = df.set_index('node')['fam'].to_dict()
node_freq_dict = df.set_index('node')['cnt'].to_dict()
unique_fams = set(node_family_dict.values())
print(f"Number of unique families: {len(unique_fams)}")


components = list(nx.connected_components(G))
remaining_components = list(nx.connected_components(G))
component_classes = []
component_nodes = []

for component in remaining_components:
    if len(component) < 4:
        continue
    families = [node_family_dict[node] for node in component if
                node in node_family_dict and not pd.isna(node_family_dict[node])]
    if families:
        # Определение класса компоненты
        most_common_class = Counter(families).most_common(1)[0][0]
        component_classes.append(most_common_class)
        component_nodes.append(component)
print(len(component_nodes))


def count_specific_kmers(sequence, target_kmers):
    """
    Считает количество указанных k-меров в последовательности нуклеотидов.

    :param sequence: строка, представляющая последовательность нуклеотидов
    :param target_kmers: список k-меров, которые нужно искать
    :return: список чисел, где каждое число — количество соответствующего k-мера
    """
    kmers_count = {kmer: 0 for kmer in target_kmers}
    k = len(target_kmers[0])
    # Проверяем, чтобы длина последовательности позволяла выделить хотя бы один k-мер
    if len(sequence) < k:
        return [0] * len(target_kmers)

    # Проходим по строке и извлекаем k-меры
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer in kmers_count:
            kmers_count[kmer] += 1

    return [kmers_count[kmer] for kmer in target_kmers]


def generate_random_kmers(k, n):
    """
    Находит все возможные k-меры и выбирает n случайных.

    :param k: длина k-мера
    :param n: количество случайных k-меров для возврата
    :return: список из n случайных k-меров
    """
    alphabet="ACGT"
    all_kmers = [''.join(kmer) for kmer in itertools.product(alphabet, repeat=k)]

    if n > len(all_kmers):
        raise ValueError(f"Количество возможных k-меров меньше, чем n")
    return random.sample(all_kmers, n)



random_kmers = generate_random_kmers(4, 100)


node2name = pd.read_csv('../../data_juncia/graph_collapse_nodes_b_juncea.txt', sep='\t')
node2name = node2name.drop_duplicates(subset=['node'], keep='first')
node2name_dict = node2name.set_index('node')['name'].to_dict()
node2name.head()


def parse_fasta(file_path):
    name2seq = {}
    with open(file_path, 'r') as fasta_file:
        current_name = None
        current_sequence = []
        for line in fasta_file:
            line = line.strip()
            if line.startswith(">"):
                if current_name:
                    name2seq[current_name] = "".join(current_sequence)
                current_name = line[1:]  # Убираем символ ">" и сохраняем имя
                current_sequence = []
            else:
                current_sequence.append(line)
        if current_name:
            name2seq[current_name] = "".join(current_sequence)  # Добавляем последнюю запись
    return name2seq

fasta_file_path = '../../data_juncia/seq_sv_big.fasta'
name2seq = parse_fasta(fasta_file_path)

# Создание набора данных на уровне узлов
X, y = [], []
for component, component_label in zip(component_nodes, component_classes):
    G_sub = G.subgraph(component).copy()
    n_nodes = len(G_sub.nodes())
    degrees = [deg for _, deg in G_sub.degree()]
    mean_degree = sum(degrees) / len(degrees) if len(degrees) > 0 else 0

    number_nodes = len(G_sub.nodes())

    max_degrees = sorted(degrees)[:3]

    freqs = sorted([node_freq_dict[node] for node in list(G_sub.nodes())], reverse=True)[:3]

    # Составим последовательость
    kmer_counts = [0] * len(random_kmers)

    for node in list(G_sub.nodes()):
        name = node2name_dict[node]
        sequence = name2seq[name]
        kmer_count = count_specific_kmers(sequence, random_kmers)

        kmer_counts = [x + y for x, y in zip(kmer_counts, kmer_count)]


    sum_len = 0
    for node in list(G_sub.nodes()):
        name = node2name_dict[node]
        sequence = name2seq[name]
        len_seq = len(sequence)
        sum_len += len_seq

    kmer_counts = [x / sum_len for x in kmer_counts]

    min_degree_count = 0
    for _, deg in G_sub.degree():
        if deg == 1:
            min_degree_count += 1

    embedding = [mean_degree, number_nodes, min_degree_count] + max_degrees + freqs + kmer_counts
    X.append(embedding)  # Добавляем эмбеддинги узла
    y.append(component_label)  # Метка компоненты


# Результат
print('Predict')
y_pred = clf.predict(X)

import re
cnt = 0
for i in range(len(component_nodes)):
    if y_pred[i] == '':
        for elem in component_nodes[i]:
            print(node2name_dict[elem])
        for elem in component_nodes[i]:
            # Регулярное выражение для поиска числа после '|'
            match = re.search(r'\|(\d+)', node2name_dict[elem])
            number = int(match.group(1))

            if number > 4500 and number < 7000:
                cnt += 1
                print(i, node2name_dict[elem])


from collections import Counter

counter = Counter(y_pred)
print(counter)
print(len(counter))








# Предсказание для кластеров после partition для brassica juncea

file_path = '../../clusters_jessica.pkl'
with open(file_path, 'rb') as f:
    matching_clusters = pickle.load(f)
print("Данные из clusters_jessica.pkl успешно загружены")

file_path = '../../families_clusters_jessica.pkl'
with open(file_path, 'rb') as f:
    families_from_clusters = pickle.load(f)
print("Данные из families_clusters_jessica.pkl успешно загружены")


X, y = [], []
matching_clusters = [x for x in matching_clusters if len(x) >= 4]
for component, component_label in zip(matching_clusters, families_from_clusters):
    G_sub = G.subgraph(component).copy()
    n_nodes = len(G_sub.nodes())
    degrees = [deg for _, deg in G_sub.degree()]
    mean_degree = sum(degrees) / len(degrees) if len(degrees) > 0 else 0

    number_nodes = len(G_sub.nodes())

    max_degrees = sorted(degrees)[:3]
    freqs = sorted([node_freq_dict[node] for node in list(G_sub.nodes())], reverse=True)[:3]

    # Составим последовательость
    kmer_counts = [0] * len(random_kmers)

    for node in list(G_sub.nodes()):
        name = node2name_dict[node]
        sequence = name2seq[name]
        kmer_count = count_specific_kmers(sequence, random_kmers)

        kmer_counts = [x + y for x, y in zip(kmer_counts, kmer_count)]


    sum_len = 0
    for node in list(G_sub.nodes()):
        name = node2name_dict[node]
        sequence = name2seq[name]
        len_seq = len(sequence)
        sum_len += len_seq

    kmer_counts = [x / sum_len for x in kmer_counts]

    min_degree_count = 0
    for _, deg in G_sub.degree():
        if deg == 1:
            min_degree_count += 1

    embedding = [mean_degree, number_nodes, min_degree_count] + max_degrees + freqs + kmer_counts
    X.append(embedding)  # Добавляем эмбеддинги узла
    y.append(component_label)  # Метка компоненты

y_pred2 = clf.predict(X)


#  Оценка того, что получилось

counter = Counter(y_pred2)
print(counter)
print(len(counter))


families_predicted = {}
for i in range(len(matching_clusters)):
    for node in matching_clusters[i]:
        families_predicted[node] = y_pred2[i]

# Семейства, которые получились при сравнении с arabidopsis thaliana
df = pd.read_csv('../../data_juncia/graph_collapse_nodes_b_juncea_full.txt', sep='\t')
df.dropna()
filtered_values = {fam: df.loc[df['fam'] == fam, 'node'].tolist() for fam in df['fam'].unique()}
filtered_dict = df.loc[df['fam'] == 'LINE'].set_index('node')['fam'].to_dict()


matches = 0
mismatches = 0
for key in families_predicted:
    if key in filtered_dict and families_predicted[key] == filtered_dict[key]:
        matches += 1
    else:
        mismatches += 1
for key in filtered_dict:
    if key not in families_predicted:
        mismatches += 1
