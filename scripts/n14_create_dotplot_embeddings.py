import numpy as np
from multiprocessing import Pool, cpu_count, Value, Lock
from tqdm import tqdm
import csv
import math
import time

import config
from scripts.n13_dotplot import dotplot

import os

wsize = 15
nmatch = 12
scatter = False


def nucleotide_fractions(seq):
    """
    Вычисляет долю каждого нуклеотида в последовательности.

    Параметры:
        seq: массив или последовательность символов, содержащая нуклеотиды
             ('A', 'T', 'G', 'C').

    Возвращает:
        Список из 4 чисел:
        [доля A, доля T, доля G, доля C].

    """
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


def create_edges_csv_start(filename):
    """
    Создаёт CSV-файл для хранения рёбер графа и записывает в него заголовок.

    Параметры:
        filename: путь к CSV-файлу.

    Что делает:
        - Создаёт директорию для файла, если она ещё не существует.
        - Открывает файл в режиме записи.
        - Записывает первую строку с названиями столбцов:
          graph_id, src, dst, edge_param.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["graph_id", "src", "dst", "edge_param"])


def append_edges_to_csv(graph_id, adj, node_index_dic, filename, wsize=15):
    """
    Добавляет рёбра одного графа в CSV-файл.

    Параметры:
        graph_id: идентификатор графа.
        adj: матрица смежности графа.
        node_index_dic: словарь соответствия исходных индексов узлов
                        новым индексам после фильтрации.
        filename: путь к CSV-файлу с рёбрами.
        wsize: размер окна, используется для нормировки веса ребра.

    Что делает:
        - Находит все ненулевые элементы матрицы смежности, то есть рёбра.
        - Для каждого ребра проверяет, не является ли один из узлов
          "исключённым" по условию:
              adj[u].sum() == adj[0][0] или adj[v].sum() == adj[0][0]
        - Если узлы подходят, записывает ребро в CSV:
          [graph_id, src, dst, adj[u][v] / wsize]

    Примечание:
        Вес ребра нормируется делением на wsize.
    """
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        src, dst = np.nonzero(adj)
        for u, v in zip(src, dst):
            if adj[u].sum() == adj[0][0] or adj[v].sum() == adj[0][0]:
                continue
            writer.writerow([graph_id, node_index_dic[u], node_index_dic[v], adj[u][v] / wsize])


def create_nodes_csv_start(filename):
    """
    Создаёт CSV-файл для хранения узлов графа и записывает заголовок.

    Параметры:
        filename: путь к CSV-файлу.

    Что делает:
        - Создаёт директорию, если её нет.
        - Создаёт CSV-файл.
        - Записывает названия столбцов:
          graph_id, node_id, feat_0, feat_1, feat_2, feat_3, feat_4
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        columns = ['graph_id', 'node_id'] + [f'feat_{i}' for i in range(5)]
        writer.writerow(columns)


def append_nodes_to_csv(graph_id, features, adj, filename):
    """
    Добавляет узлы одного графа в CSV-файл и формирует словарь новых индексов.

    Параметры:
        graph_id: идентификатор графа.
        features: массив признаков узлов.
        adj: матрица смежности графа.
        filename: путь к CSV-файлу с узлами.

    Возвращает:
        node_index_dic: словарь, где:
            ключ   — исходный индекс узла,
            значение — новый индекс узла в выходном CSV.

    Что делает:
        - Проходит по всем узлам графа.
        - Для каждого узла сохраняет соответствие старого индекса новому.
        - Пропускает узлы, удовлетворяющие условию:
              adj[node_id].sum() == adj[0][0]
        - Для остальных узлов формирует признаки:
              identical_fractions(features[node_index]) + [node_id / num_nodes]
        - Записывает строку в CSV:
              [graph_id, node_index] + feature

    Примечание:
        Сейчас вместо реальных долей нуклеотидов используется
        identical_fractions(...), то есть фиксированный вектор
        [0.25, 0.25, 0.25, 0.25].
    """
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


def read_fasta(fasta_path):
    header = None
    seq_parts = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(seq_parts)
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)

        if header is not None:
            yield header, ''.join(seq_parts)


def chunk_list(data, n_chunks):
    """
    Делит список data на n_chunks примерно равных частей.
    """
    chunk_size = math.ceil(len(data) / n_chunks)
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def process_fasta_chunk(args):
    chunk_id, records, temp_dir, wsize, nmatch, scatter = args

    nodes_part = os.path.join(temp_dir, f"nodes_part_{chunk_id}.csv")
    edges_part = os.path.join(temp_dir, f"edges_part_{chunk_id}.csv")

    create_nodes_csv_start(nodes_part)
    create_edges_csv_start(edges_part)

    for header, sequence in records:
        graph_id = header.split('\t')[0]

        features, matrix = dotplot(
            sequence,
            sequence,
            wsize=wsize,
            nmatch=nmatch,
            scatter=scatter
        )

        node_index_dic = append_nodes_to_csv(
            graph_id=graph_id,
            features=features,
            adj=matrix,
            filename=nodes_part
        )

        append_edges_to_csv(
            graph_id=graph_id,
            adj=matrix,
            node_index_dic=node_index_dic,
            filename=edges_part,
            wsize=wsize
        )

        with counter_lock:
            counter.value += 1

    return nodes_part, edges_part


def merge_csv_files(part_files, output_file):
    """
    Склеивает несколько CSV-файлов в один.
    Заголовок берётся только из первого файла.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    first_file = True
    with open(output_file, 'w', newline='') as fout:
        writer = csv.writer(fout)

        for part_file in part_files:
            with open(part_file, 'r', newline='') as fin:
                reader = csv.reader(fin)
                header = next(reader)

                if first_file:
                    writer.writerow(header)
                    first_file = False

                for row in reader:
                    writer.writerow(row)


counter = None
counter_lock = None


def init_worker(c, l):
    global counter, counter_lock
    counter = c
    counter_lock = l


def create_and_write_graphs_from_fasta_parallel(
    fasta_path,
    file_edges,
    file_nodes,
    n_processes=None,
    temp_dir=None
):
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)

    if temp_dir is None:
        temp_dir = os.path.join(os.path.dirname(file_nodes), "tmp_parts")

    os.makedirs(temp_dir, exist_ok=True)

    records = list(read_fasta(fasta_path))
    total = len(records)

    if total == 0:
        raise ValueError("FASTA-файл пустой")

    chunks = chunk_list(records, n_processes)

    counter = Value('i', 0)
    lock = Lock()

    worker_args = [
        (i, chunk, temp_dir, wsize, nmatch, scatter)
        for i, chunk in enumerate(chunks)
        if chunk
    ]

    with Pool(
        processes=n_processes,
        initializer=init_worker,
        initargs=(counter, lock)
    ) as pool:
        result = pool.map_async(process_fasta_chunk, worker_args)

        with tqdm(total=total, desc="Processed sequences") as pbar:
            last = 0
            while not result.ready():
                time.sleep(0.2)
                with lock:
                    current = counter.value
                pbar.update(current - last)
                last = current

            with lock:
                current = counter.value
            pbar.update(current - last)

        results = result.get()

    node_parts = [x[0] for x in results]
    edge_parts = [x[1] for x in results]

    merge_csv_files(node_parts, file_nodes)
    merge_csv_files(edge_parts, file_edges)


if __name__ == '__main__':
    fasta_path = f"{config.DIR_REPBASE_PROCESSED}/all_sequences_filtered_01.fasta"

    file_edges = f"{config.DIR_DOTPLOTS}/edges.csv"
    file_nodes = f"{config.DIR_DOTPLOTS}/nodes.csv"

    create_and_write_graphs_from_fasta_parallel(
        fasta_path=fasta_path,
        file_edges=file_edges,
        file_nodes=file_nodes
    )
