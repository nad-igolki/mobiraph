#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import gc
import glob
import math
import os
import time
from multiprocessing import Pool, cpu_count, Value, Lock

import numpy as np
from tqdm import tqdm

from scripts.n13_dotplot import dotplot


counter = None
counter_lock = None


def init_worker(c, l):
    global counter, counter_lock
    counter = c
    counter_lock = l


def load_processed_ids(log_path):
    if not os.path.exists(log_path):
        return set()

    processed = set()
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                processed.add(line)
    return processed


def append_processed_id(log_path, graph_id, lock=None):
    if lock is not None:
        with lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{graph_id}\n")
    else:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{graph_id}\n")


def nucleotide_fractions(seq):
    """
    Вычисляет долю каждого нуклеотида в последовательности.
    Возвращает [A, T, G, C].
    """
    total = len(seq)
    if total == 0:
        return [0.0, 0.0, 0.0, 0.0]

    return [
        np.count_nonzero(seq == "A") / total,
        np.count_nonzero(seq == "T") / total,
        np.count_nonzero(seq == "G") / total,
        np.count_nonzero(seq == "C") / total,
    ]


def identical_fractions(_seq):
    return [0.25, 0.25, 0.25, 0.25]


def create_edges_csv_start(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["graph_id", "src", "dst", "edge_param"])


def append_edges_to_csv(graph_id, adj, node_index_dic, filename, wsize=15):
    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        src, dst = np.nonzero(adj)

        for u, v in zip(src, dst):
            if adj[u].sum() == adj[0][0] or adj[v].sum() == adj[0][0]:
                continue

            if u not in node_index_dic or v not in node_index_dic:
                continue

            writer.writerow([
                graph_id,
                node_index_dic[u],
                node_index_dic[v],
                adj[u][v] / wsize
            ])


def create_nodes_csv_start(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        columns = ["graph_id", "node_id"] + [f"feat_{i}" for i in range(5)]
        writer.writerow(columns)


def append_nodes_to_csv(graph_id, features, adj, filename):
    """
    Добавляет узлы одного графа в CSV-файл и возвращает словарь:
    исходный индекс узла -> новый индекс узла.
    """
    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        num_nodes = features.shape[0]

        node_index = 0
        node_index_dic = {}

        for node_id in range(num_nodes):
            if adj[node_id].sum() == adj[0][0]:
                continue

            node_index_dic[node_id] = node_index
            feature = identical_fractions(features[node_id]) + [node_id / num_nodes]
            row = [graph_id, node_index] + feature
            writer.writerow(row)
            node_index += 1

    return node_index_dic


def read_fasta(fasta_path):
    header = None
    seq_parts = []

    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)

        if header is not None:
            yield header, "".join(seq_parts)


def chunk_list(data, n_chunks):
    """
    Делит список data на n_chunks примерно равных частей.
    """
    if not data:
        return []

    chunk_size = math.ceil(len(data) / n_chunks)
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def process_fasta_chunk(args):
    chunk_id, records, temp_dir, wsize, nmatch, scatter, processed_log, run_id = args

    nodes_part = os.path.join(temp_dir, f"nodes_part_{run_id}_{chunk_id}.csv")
    edges_part = os.path.join(temp_dir, f"edges_part_{run_id}_{chunk_id}.csv")

    create_nodes_csv_start(nodes_part)
    create_edges_csv_start(edges_part)

    for header, sequence in records:
        sequence = sequence.upper()
        graph_id = header.split("\t")[0]

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

        append_processed_id(processed_log, graph_id, counter_lock)

        with counter_lock:
            counter.value += 1

        del sequence, features, matrix, node_index_dic
        gc.collect()

    return nodes_part, edges_part


def merge_csv_files(part_files, output_file):
    """
    Склеивает несколько CSV-файлов в один.
    Заголовок берётся только из первого файла.
    """
    if not part_files:
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    first_file = True
    with open(output_file, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)

        for part_file in part_files:
            with open(part_file, "r", newline="", encoding="utf-8") as fin:
                reader = csv.reader(fin)

                try:
                    header = next(reader)
                except StopIteration:
                    continue

                if first_file:
                    writer.writerow(header)
                    first_file = False

                for row in reader:
                    writer.writerow(row)


def create_and_write_graphs_from_fasta_parallel(
    fasta_path,
    file_edges,
    file_nodes,
    wsize=15,
    nmatch=12,
    scatter=False,
    n_processes=None,
    temp_dir=None
):
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)

    if temp_dir is None:
        temp_dir = os.path.join(os.path.dirname(file_nodes), "tmp_parts")

    os.makedirs(temp_dir, exist_ok=True)

    processed_log = os.path.join(os.path.dirname(file_nodes), "processed_sequences.log")
    processed_ids = load_processed_ids(processed_log)

    records = list(read_fasta(fasta_path))
    total_all = len(records)

    if total_all == 0:
        raise ValueError("FASTA-файл пустой")

    records = [
        (header, seq)
        for header, seq in records
        if header.split("\t")[0] not in processed_ids
    ]

    total = len(records)
    print(f"Всего последовательностей: {total_all}")
    print(f"Уже обработано: {len(processed_ids)}")
    print(f"Осталось обработать: {total}")

    if total == 0:
        print("Все последовательности уже обработаны.")
        node_parts = sorted(glob.glob(os.path.join(temp_dir, "nodes_part_*.csv")))
        edge_parts = sorted(glob.glob(os.path.join(temp_dir, "edges_part_*.csv")))

        if node_parts:
            merge_csv_files(node_parts, file_nodes)
        if edge_parts:
            merge_csv_files(edge_parts, file_edges)
        return

    chunks = chunk_list(records, n_processes)
    del records
    gc.collect()

    shared_counter = Value("i", 0)
    lock = Lock()

    run_id = int(time.time())

    worker_args = [
        (i, chunk, temp_dir, wsize, nmatch, scatter, processed_log, run_id)
        for i, chunk in enumerate(chunks)
        if chunk
    ]

    with Pool(
        processes=n_processes,
        initializer=init_worker,
        initargs=(shared_counter, lock)
    ) as pool:
        result = pool.map_async(process_fasta_chunk, worker_args)

        with tqdm(total=total, desc="Processed sequences") as pbar:
            last = 0
            while not result.ready():
                time.sleep(0.2)
                with lock:
                    current = shared_counter.value
                pbar.update(current - last)
                last = current

            with lock:
                current = shared_counter.value
            pbar.update(current - last)

        result.get()

    node_parts = sorted(glob.glob(os.path.join(temp_dir, "nodes_part_*.csv")))
    edge_parts = sorted(glob.glob(os.path.join(temp_dir, "edges_part_*.csv")))

    merge_csv_files(node_parts, file_nodes)
    merge_csv_files(edge_parts, file_edges)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Параллельная обработка FASTA и генерация CSV-файлов узлов и рёбер."
    )

    parser.add_argument(
        "--fasta",
        required=True,
        help="Путь к входному FASTA-файлу."
    )
    parser.add_argument(
        "--nodes",
        required=True,
        help="Путь к выходному CSV-файлу узлов."
    )
    parser.add_argument(
        "--edges",
        required=True,
        help="Путь к выходному CSV-файлу рёбер."
    )
    parser.add_argument(
        "--wsize",
        type=int,
        default=15,
        help="Размер окна для dotplot. По умолчанию: 15."
    )
    parser.add_argument(
        "--nmatch",
        type=int,
        default=12,
        help="Минимум совпадений для dotplot. По умолчанию: 12."
    )
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="Включить scatter-режим для dotplot."
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Количество процессов. По умолчанию: cpu_count() - 1."
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Каталог для временных CSV-частей."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    create_and_write_graphs_from_fasta_parallel(
        fasta_path=args.fasta,
        file_edges=args.edges,
        file_nodes=args.nodes,
        wsize=args.wsize,
        nmatch=args.nmatch,
        scatter=args.scatter,
        n_processes=args.processes,
        temp_dir=args.temp_dir
    )


if __name__ == "__main__":
    main()