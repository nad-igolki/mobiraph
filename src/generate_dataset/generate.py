from utils.fast_dot_plot_dna import dotplot, reverse_complement
import random
import numpy as np

def generate_LTR(wsize=15, nmatch=12, fill_diagonal_zero=False, scatter=False):
    length = random.randint(1000, 5000)
    nucleotides = ['A', 'T', 'C', 'G']
    sequence = ''.join(random.choices(nucleotides, k=length))
    length_rp_part = random.randint(100, 500)

    start_part = sequence[:length_rp_part]

    sequence = sequence[:-length_rp_part] + start_part
    features, matrix = dotplot(sequence, sequence, wsize=wsize, nmatch=nmatch, scatter=scatter)

    if fill_diagonal_zero:
        np.fill_diagonal(matrix, 0)

    return features, matrix, sequence


def generate_TIR(wsize=15, nmatch=12, fill_diagonal_zero=False, scatter=False):
    length = random.randint(1000, 5000)
    nucleotides = ['A', 'T', 'C', 'G']
    sequence = ''.join(random.choices(nucleotides, k=length))
    length_rp_part = random.randint(100, 500)

    start_part = sequence[:length_rp_part]
    palindromic_end = reverse_complement(start_part)

    sequence = sequence[:-length_rp_part] + palindromic_end
    features, matrix = dotplot(sequence, sequence, wsize=wsize, nmatch=nmatch, scatter=scatter)

    if fill_diagonal_zero:
        np.fill_diagonal(matrix, 0)

    return features, matrix, sequence


def generate_NO(wsize=15, nmatch=12, fill_diagonal_zero=False, scatter=False):
    length = random.randint(1000, 5000)
    nucleotides = ['A', 'T', 'C', 'G']
    sequence = ''.join(random.choices(nucleotides, k=length))
    features, matrix = dotplot(sequence, sequence, wsize=wsize, nmatch=nmatch, scatter=scatter)

    if fill_diagonal_zero:
        np.fill_diagonal(matrix, 0)

    return features, matrix, sequence


def generate_INNER(wsize=15, nmatch=12, fill_diagonal_zero=False, scatter=False):
    length = random.randint(1000, 5000)
    nucleotides = ['A', 'T', 'C', 'G']
    sequence = ''.join(random.choices(nucleotides, k=length))
    length_rp_part = random.randint(100, 400)

    repeat_seq = ''.join(random.choices(nucleotides, k=length_rp_part))

    insert_pos1 = random.randint(0, length - 2 * length_rp_part)

    available_range = list(range(0, insert_pos1 - length_rp_part + 1)) + \
                      list(range(insert_pos1 + length_rp_part, length - length_rp_part + 1))

    insert_pos2 = random.choice(available_range)

    sequence = list(sequence)
    sequence[insert_pos1:insert_pos1 + length_rp_part] = repeat_seq
    sequence = ''.join(sequence)
    sequence = list(sequence)
    sequence[insert_pos2:insert_pos2 + length_rp_part] = repeat_seq
    sequence = ''.join(sequence)

    sequence = ''.join(sequence)

    features, matrix = dotplot(sequence, sequence, wsize=wsize, nmatch=nmatch, scatter=scatter)

    if fill_diagonal_zero:
        np.fill_diagonal(matrix, 0)

    return features, matrix, sequence


