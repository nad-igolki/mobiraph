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


import itertools
import random

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

# Генерируем рандомные k-меры
random_kmers = generate_random_kmers(4, 100)

node2name = pd.read_csv('data_arob/graph_collapse_nodes.txt', sep='\t')
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

fasta_file_path = 'data_arob/sv_pangen_seq_sv_big.fasta'
name2seq = parse_fasta(fasta_file_path)

# Создание набора данных на уровне узлов
X, y = [], []

for component, component_label in zip(component_nodes, component_classes):
    G_sub = G_not_full.subgraph(component).copy()
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
        sequecnce = name2seq[name]
        kmer_count = count_specific_kmers(sequecnce, random_kmers)

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
