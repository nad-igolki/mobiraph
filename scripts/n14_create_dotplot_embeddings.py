#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import gc
import math
import os
import shutil
import time
from multiprocessing import Pool, cpu_count, Value, Lock
import config

import numpy as np
from tqdm import tqdm

from scripts.n13_dotplot import dotplot


counter = None
counter_lock = None
io_lock = None


def init_worker(c, c_lock, shared_io_lock):
    global counter, counter_lock, io_lock
    counter = c
    counter_lock = c_lock
    io_lock = shared_io_lock


def load_finished_ids(log_path):
    """
    Читает лог статусов и возвращает только те graph_id,
    у которых последний статус == finished.
    """
    if not os.path.exists(log_path):
        return set()

    statuses = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                continue

            graph_id, status = parts
            statuses[graph_id] = status

    return {graph_id for graph_id, status in statuses.items() if status == "finished"}


def append_status(log_path, graph_id, status, lock=None):
    """
    Пишет в лог строку вида:
    graph_id<TAB>started
    graph_id<TAB>finished

    Лог append-only: актуальным считается последний статус graph_id.
    """
    if lock is not None:
        with lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{graph_id}\t{status}\n")
    else:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{graph_id}\t{status}\n")


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


def create_nodes_csv_start(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        columns = ["graph_id", "node_id"] + [f"feat_{i}" for i in range(5)]
        writer.writerow(columns)


def append_graph_to_csvs(graph_id, adj, filename_edges, filename_nodes, features, wsize=15, lock=None):
    """
    Сразу дозаписывает результат одного graph_id в финальные CSV.
    Это позволяет безопаснее переживать перезапуски:
    finished ставим только после записи данных.
    """
    num_nodes = features.shape[0]

    row_sums = adj.sum(axis=1)
    diag = np.diag(adj)
    active_mask = row_sums != diag
    active_nodes = np.flatnonzero(active_mask)

    new_index = np.full(num_nodes, -1, dtype=np.int64)
    new_index[active_nodes] = np.arange(active_nodes.shape[0])

    node_rows = []
    for old_id in active_nodes:
        feature = identical_fractions(features[old_id]) + [old_id / num_nodes]
        node_rows.append([graph_id, new_index[old_id], *feature])

    src, dst = np.nonzero(np.triu(adj, k=1))
    edge_mask = active_mask[src] & active_mask[dst]
    src = src[edge_mask]
    dst = dst[edge_mask]

    new_src = new_index[src]
    new_dst = new_index[dst]
    weights = adj[src, dst] / wsize

    edge_rows = np.column_stack([
        np.full(src.shape[0], graph_id),
        new_src,
        new_dst,
        weights
    ]).tolist()

    if lock is not None:
        with lock:
            with open(filename_nodes, mode="a", newline="", encoding="utf-8") as f_nodes:
                writer = csv.writer(f_nodes)
                writer.writerows(node_rows)

            with open(filename_edges, mode="a", newline="", encoding="utf-8") as f_edges:
                writer = csv.writer(f_edges)
                writer.writerows(edge_rows)
    else:
        with open(filename_nodes, mode="a", newline="", encoding="utf-8") as f_nodes:
            writer = csv.writer(f_nodes)
            writer.writerows(node_rows)

        with open(filename_edges, mode="a", newline="", encoding="utf-8") as f_edges:
            writer = csv.writer(f_edges)
            writer.writerows(edge_rows)


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


def cleanup_temp_dir(temp_dir):
    """
    При новом запуске удаляем временные файлы прошлого запуска.
    """
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)


def prepare_output_files(file_nodes, file_edges, processed_log, reset_outputs=False):
    """
    Если reset_outputs=True — пересоздаём итоговые CSV.
    Иначе создаём только если их ещё нет.
    """
    if reset_outputs or not os.path.exists(file_nodes):
        create_nodes_csv_start(file_nodes)

    if reset_outputs or not os.path.exists(file_edges):
        create_edges_csv_start(file_edges)

    if reset_outputs and os.path.exists(processed_log):
        os.remove(processed_log)


def split_fasta_to_chunk_fastas(fasta_path, temp_dir, n_chunks, finished_ids):
    """
    Потоково читает входной FASTA и раскладывает необработанные записи
    по chunk-*.fasta (round-robin), не держа всё в памяти.

    Возвращает:
      - chunk_files: список непустых chunk fasta
      - total_all: всего последовательностей во входном файле
      - total_remaining: сколько реально нужно обработать
    """
    if n_chunks <= 0:
        raise ValueError("n_chunks должен быть > 0")

    chunk_paths = [os.path.join(temp_dir, f"chunk_{i}.fasta") for i in range(n_chunks)]
    chunk_counts = [0] * n_chunks

    handles = [open(path, "w", encoding="utf-8") for path in chunk_paths]

    total_all = 0
    total_remaining = 0
    chunk_idx = 0

    try:
        for header, seq in read_fasta(fasta_path):
            if len(seq) > 20000:
                continue
            total_all += 1
            graph_id = header.split("\t")[0]

            if graph_id in finished_ids:
                continue

            h = handles[chunk_idx]
            h.write(f">{header}\n{seq}\n")

            chunk_counts[chunk_idx] += 1
            total_remaining += 1
            chunk_idx = (chunk_idx + 1) % n_chunks
    finally:
        for h in handles:
            h.close()

    non_empty_chunk_files = []
    for path, count in zip(chunk_paths, chunk_counts):
        if count > 0:
            non_empty_chunk_files.append(path)
        else:
            os.remove(path)

    return non_empty_chunk_files, total_all, total_remaining


def process_fasta_chunk(args):
    chunk_id, chunk_fasta, file_nodes, file_edges, wsize, nmatch, scatter, processed_log = args

    for header, sequence in read_fasta(chunk_fasta):
        sequence = sequence.upper()
        graph_id = header.split("\t")[0]

        append_status(processed_log, graph_id, "started", io_lock)

        features, matrix = dotplot(
            sequence,
            sequence,
            wsize=wsize,
            nmatch=nmatch,
            scatter=scatter
        )

        append_graph_to_csvs(
            graph_id=graph_id,
            features=features,
            adj=matrix,
            filename_nodes=file_nodes,
            filename_edges=file_edges,
            wsize=wsize,
            lock=io_lock
        )

        append_status(processed_log, graph_id, "finished", io_lock)

        with counter_lock:
            counter.value += 1

        del sequence, features, matrix
        gc.collect()

    return chunk_id


def create_and_write_graphs_from_fasta_parallel(
    fasta_path,
    output_dir,
    wsize=15,
    nmatch=12,
    scatter=False,
    n_processes=None,
    reset_outputs=False,
):
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)

    os.makedirs(output_dir, exist_ok=True)

    file_nodes = os.path.join(output_dir, "nodes.csv")
    file_edges = os.path.join(output_dir, "edges.csv")
    temp_dir = os.path.join(output_dir, "tmp_parts")
    processed_log = os.path.join(output_dir, "processed_sequences.log")

    prepare_output_files(file_nodes, file_edges, processed_log, reset_outputs=reset_outputs)

    finished_ids = load_finished_ids(processed_log)

    # Удаляем времянку прошлого запуска и создаём новую
    cleanup_temp_dir(temp_dir)

    chunk_files, total_all, total = split_fasta_to_chunk_fastas(
        fasta_path=fasta_path,
        temp_dir=temp_dir,
        n_chunks=n_processes,
        finished_ids=finished_ids
    )

    if total_all == 0:
        raise ValueError("FASTA-файл пустой")

    print(f"Всего последовательностей: {total_all}")
    print(f"Уже завершено по логу: {len(finished_ids)}")
    print(f"Осталось обработать: {total}")
    print(f"Результаты будут сохранены в: {output_dir}")

    if total == 0:
        print("Все последовательности уже обработаны.")
        return

    shared_counter = Value("i", 0)
    progress_lock = Lock()
    shared_io_lock = Lock()

    worker_args = [
        (i, chunk_fasta, file_nodes, file_edges, wsize, nmatch, scatter, processed_log)
        for i, chunk_fasta in enumerate(chunk_files)
    ]

    with Pool(
        processes=min(n_processes, len(worker_args)),
        initializer=init_worker,
        initargs=(shared_counter, progress_lock, shared_io_lock)
    ) as pool:
        result = pool.map_async(process_fasta_chunk, worker_args)

        with tqdm(total=total, desc="Processed sequences") as pbar:
            last = 0
            while not result.ready():
                time.sleep(0.2)
                with progress_lock:
                    current = shared_counter.value
                pbar.update(current - last)
                last = current

            with progress_lock:
                current = shared_counter.value
            pbar.update(current - last)

        result.get()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Параллельная обработка FASTA и генерация CSV-файлов узлов и рёбер."
    )

    parser.add_argument(
        "--fasta",
        default=f"{config.DIR_REPBASE_PROCESSED}/all_sequences_filtered_01.fasta",
        help="Путь к входному FASTA-файлу."
    )
    parser.add_argument(
        "--output-dir",
        default=f"{config.DIR_DOTPLOTS}",
        help="Каталог, куда будут сохранены nodes.csv, edges.csv, tmp_parts и processed_sequences.log."
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
        required=True,
        help="Количество процессов."
    )
    parser.add_argument(
        "--reset-outputs",
        action="store_true",
        help="Пересоздать nodes.csv и edges.csv с нуля."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    create_and_write_graphs_from_fasta_parallel(
        fasta_path=args.fasta,
        output_dir=args.output_dir,
        wsize=args.wsize,
        nmatch=args.nmatch,
        scatter=args.scatter,
        n_processes=args.processes,
        reset_outputs=args.reset_outputs,
    )


if __name__ == "__main__":
    main()