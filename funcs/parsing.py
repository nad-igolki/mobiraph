import itertools
import random

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
