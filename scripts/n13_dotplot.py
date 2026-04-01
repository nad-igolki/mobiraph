import numpy as np
import matplotlib.pyplot as plt


def seq2mx(seq, wsize):
    seq = np.array(list(seq))
    matrix_seq = np.array([seq[i:i + wsize] for i in range(len(seq) - wsize + 1)])
    return matrix_seq

def mxComp(mx1, mx2, nmatch):
    mx_res = np.zeros((len(mx1), len(mx2)), dtype=int)
    for s in ["A", "C", "G", "T"]:
        mx_res += ((mx1 == s).astype(int)) @ ((mx2 == s).astype(int)).T

    mx_res[mx_res < nmatch] = 0

    return mx_res

def reverse_complement(seq):
    complement = str.maketrans("ACGT", "TGCA")
    return seq.translate(complement)[::-1]

def dotplot(seq1, seq2, wsize=15, nmatch=12, scatter=False):
    if wsize < nmatch:
        raise ValueError("wsize must be larger than nmatch")

    seq2_rc = reverse_complement("".join(seq2))

    mx1 = seq2mx(seq1, wsize)
    mx2 = seq2mx(seq2, wsize)
    mx_rc = seq2mx(seq2_rc, wsize)

    result = mxComp(mx1, mx2, nmatch)
    result_rc =  mxComp(mx1, mx_rc, nmatch)

    max_result = np.maximum(result, result_rc)

    if scatter:
        rows, cols = np.nonzero(result)
        rows_rc, cols_rc = np.nonzero(result_rc)
        cols_rc = len(seq2) - cols_rc - wsize + 2

        plt.scatter(cols, rows, c='black', s=0.5)
        plt.scatter(cols_rc, rows_rc, c='red', s=0.5)
        plt.gca().set_aspect('equal')
        plt.title(f"({wsize}, {nmatch})")
        plt.grid(True)
        plt.show()

    return mx1, max_result

# helitron = "aaaaaaaatttgtttctaaaagattgattttttaagttttctatgtaatatttattggttagtattggtgaattgtaattttcaagaaaaatagttaattctcattggtttagagtagggatgtcaaaatgggtaacccaactcaactcataatcaaatgagtttaaggttaaatgagttatgggttgacccaactcattttgttaaataggttgggtctacctataactcatttaatatgggttaacccatttaaataataatttaattaattattattataaaaataataaattaataatgattcattatcatcaaacttaggatatttacggattccactttttacggatttacgtttttgacgagaaaatcatgggtttacgtttttggcgggaaaatctcgggtttacgtttttggcggaaaaatcacggatttacgtttttggcgggaaaatcacggatttacgtttttggcgggaaaatcacggttttttgttttttggcgggaaaattacgagttaacgtttttggcgggaaaattacgaatttacgtttttggcgggaaaatcacgggtttacgtttttggcgggaaaatcacaggatttacatttttggcgggaaaatcttgggtttacgtttttggcgagaaaatcttgggtttacgttttttgcaggaaaatcacgggtttacttttttggcgggaaaatcacgggtttatgttttttggtggaaaaattacgagtttactttttctcaatttcatcgattgtatatttaagaaatttggaaaaatattaattttattaaattggtttagatgtgttggttaaacttaaattgacattggtttagagattttagttggtttaattcaattttacaaaacttattgggttaattgggtaaaccattaaaaccattaaccattacaacccaactcattttactcatcaaaccaattgactcatcaactcatttgacccatcaactcatttgagtcaaaaattttaactcattagggttcatggattgagttgagttgagttgaccatgaattttgacccattttgacacccctagtttagagttatagaaaactgtaaaacactaaaaataatacatttataatcaacatttaatatgttttcttaatatgtgtgtttttctaaacaatcaaacaaaaatgaacggaggaa"
# helitron = helitron.upper()
# features, graph = dotplot(helitron, helitron, wsize=15, nmatch=12, scatter=True)