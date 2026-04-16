import numpy as np


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
        import matplotlib.pyplot as plt
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
# features, graph = dotplot(helitron, helitron, wsize=15, nmatch=10, scatter=True)
# g = "tgttgcagactatgcaaaccttcattagttttataagtttaatatccgccttagtttgatctacccaaagtaggcgcgcgcaataaccactgtttgtgctaattgtcatcacgcgcgtaactatttgttcccttggtcagaactcctagataaaggggctgatcattctgatactctctctcctgccccaaagtattctgttgtcccctattcactgaataaagtccttattcatctactactcacatactgtctggctggtggttcacacaatgccgccaccaggagacacaaca"
# g = g.upper()
# features, graph = dotplot(g, g, wsize=15, nmatch=12, scatter=True)

# sequence = (
#     "tgtaaaccctgaataatgctttaaggaattttagctagttctttatttcataagaataatggtaaggatctataggactt"
#     "tagttaatcaagaaaaaaatattaaagttatttttagtatgtgtctaagtgttgaagacatgttccttttgaatcaatag"
#     "gtgtgaattatttcaccataataatacctattggattatacaatgatgtgaatttctcgcttattgaatatcataaatga"
#     "aaataaatacatgatcaaattaagagaggaataaatacacataaaatattattgcgtcatgctaggattattttgtttgt"
#     "caaaactcagaattgaatttgtttgaaaaaaattgaaattgaaacttgaaaataaaatgaaataaagaaaaggaattgaa"
#     "aaacagaaaataaaagaaatcgtcgtttgggccgaatccactacatcggcccacctcaccccacacctgcgtggcccgcg"
#     "atccgtgcttgggtcaccgacacgtgggccctctgtgtcagcctcttcgctggctgtgcgggttgttgtgggtggctgac"
#     "ttgtggagccgatttgtcagccgagtttttctcttccccaacaaacttccgccatccgtgctccatggtcagctcaacag"
#     "aacctccataaatccatgcgccacaatggctggcaaactcaaacgccgataccgccatgagttgttacatcctccctacc"
#     "catcgcacgctgcagttgacgaggaactctgcctttgcggactataaaacggagctcgtacccccctttttcctggccat"
#     "cttcacagagtctgcagatggaggaatcgggccgccattggggtattccacttccacccacgctgctggaacccgctcga"
#     "aggctctggggttcacgcaagtccactgaaaggcagacaaggcagaatctgcacgcgaagatgacatgtgcttcatgaat"
#     "ccctcatcgacggagatccgccgccgtgggaccgctatggcgtgtggatggatactgccatcccaaatcaaggtatggta"
#     "gccactatcgtgttcgactcctgttcagctgcgcgtagaacatagcggcatgggcaaaagagcactggtacgcggaataa"
#     "tcgctcgccggtggtgctctgtcgcggcacggcgtgctgtcaagttggtgggcgccgtagggaagaagacgcgggtctgg"
#     "gagcttcgtcggttaccctggggcgtgttcgatgcggtgctcgtttccatccttctgtccatgctcgtgccgcctgggca"
#     "ctcgtgctcgccgtcgtcatgtcgacagggcgccgcctcctagtttctgtcatggctctggtccgcacctcaccaggatc"
#     "cagccccaagctgccagcaacctccagcgttgctcatcgctcaccgcacactgccttgcgcccccgcccagttctgctct"
#     "cccttcgtcgctttgcggctctgccattgtttggacgtccggcctctggctttggcggccgccacccgccatctccacct"
#     "tgctgccggttcgatctcagagaaggcactcagagcatcaccgctataccagggaagttgtgttcaagactcaccacgcc"
#     "tggggcgttcttccgtggccggaatttctcgtcagtgcggggtttgtgccgtcgtccgccccgcaccatggtcatgggtg"
#     "ccgctgcaccaagcaccgttaacgattcccccatggcgctctccatcgaccgtaccatgtgtattcctctttgaattagc"
#     "tttcggacaccaattgtgggggggaaggtatcaacggcgtggctccaccgtggtcctgtgtgggcgccgccgtgctggag"
#     "ctgcgtcatcgctgtcctggtcgggattagggtgttgatcgggcaccgttcgatctcaatctaaggctctggattaaaac"
#     "ctagaatatcgtttcgggattttggtctcaatcgtccgatcagatcgtggggctacgattaaatctgcacccccctcgtc"
#     "cccatgttatggtggccgttgatctcaggatcggcggctcgggtgagttggctggtttggatctaatctcggcccccgat"
#     "atcggatcgggcggccaggatgctcgtataccccttcggcctggcgatctttctaaagagcccctcggcttgttccaaat"
#     "aaacccgcagtccacactgcgggagttctgagtcttggtgtcttttgcgcagaaccccctgggtttctattaaatagtgg"
#     "ccgcagtccaagtaactcagaaatttattaaatttaatacagaaattctttttaatgcaaaataatgtcccaaacttaag"
#     "aaattcataacttggtgaatttagctccaaattgattcattccagtggcattaattttgtttaagtattatttattacct"
#     "agtacctttgtttagccacaaaacttgaattaaaattgttcatttggattaaactattctaagcactaaataattttgga"
#     "aattcataacttaataaccgtagttccgaatttagcggttctcgaacccacgatctcgtaacgacgcgtagattattatt"
#     "atgtatcttgttcttatgtttggtgtgatgttaattttgcctataccatatatgtctgtattgctacgtttagcggggag"
#     "gacgagtcacctgaagatcatcctggtacctggaatctcaagtcccaggcaagttgtgcccttgatcacttctttttacc"
#     "cactcatgttctaattaatcataatgatctgcataggttaattttgatgggacccaataggttaccctagtttgattatc"
#     "tttataccttgtttaccactgaactttttgggtagtacttgctagtgctttatgtggttttgggtatgaagatacattat"
#     "tcatgatcacacttttattatctgtttattatcactgttcatgataagatcattatgttaattggaacatggagcgacca"
#     "cccgggaaaacagtgctaccacaagggtataatgggacgccctcggctgattaattaggaaagctagtggaggactacct"
#     "tacccgaaaggggcaagggcagtaggggagtggtcagtgtagggaggccctctggaggattttgctgcgatggcggtcct"
#     "gcaagggattcctgcattggagcttcctataaactgtagcgggttttctgaagctagtggaactttgtaaaggcctcgta"
#     "gtgttaccctgcctcgcctcctcggtagaggtgtatgggaagtcgcgatcccttggcagatgggtaacatgacttgtggg"
#     "taaagatgcgcaacctctgcagagtgtaaaactggtatactagccgtgctcacggtcatgagcagctcggaccctcacat"
#     "gattaatttatggaacttaaattcaatttgtcatatgcattgcatcgcaggtgatgttgttacttctgttctactattta"
#     "attgggctggtatttacttatacttagtaattgctaataaaattttgaccaactttaaaagcaatgctcagcttcaacca"
#     "tcttctttggtaagccttacacttcacgtgagctcccgcctttggcgagttcatgcacattattccccacaacttgttga"
#     "gcgatgaacgtatgtgagctcactcttgctgtctcacacccccccacacaggtcaagaacaggtaccacaggatgaggcg"
#     "catgaaggatgctgtgacgagttcgtgagaggtctaggtcgtcgtctcccagtcaacttcgggttgctggaccgttgtct"
#     "ccatataatgtaattatttatttattttgtacagaactccgattatatagtaaagatgtgacattcgatcctgtgccatg"
#     "attcatcatatgtgtgagacttggtcccagcacacctggtgattatgttcgcgcccgggtcttggtgccccgaaacccgg"
#     "gtgtgacagaagtggtatcagaggaatgttgactgtaggacgaaacctagatagaactggacaacccttatctattcacc"
#     "tctgctactctgattcttttctaaactgatcttaatcttttctcatctatttcgctttactctgattattcttatctttc"
#     "tttctaaagacaaatgtggatttcacactttgaaatcttgtgcctaaagtgacctttaggaataggcgacctactcttag"
#     "gaacaaaatcaaaactatttttgtgaatatttgtacgcttgagtgtttgttcttatgatacttgtctgatttggatcttt"
#     "ggttgagtgtgatgggttgtggagtaatgtccacaattacatctacatatacgtataggcataaatataaaaaaatcata"
#     "agatgactaaacaaacttagaatattccctattaaaaatatatctagtttaatagatccatcttatcttaaaagattctc"
#     "tcttatcttaatagattcatcttatcttgaaaaaatcatcttatcctaaccacttatttattttatcctaaggtaacaac"
#     "catgatctaatcaatgaaatcttatcaagtctatctagtcatatccaatctaatctagattttatctaatctattacgat"
#     "ctaatctagagtgtgttacttaatgtgggtacaaaggatttggccatactcctaaataactcaagcctaaattggtgact"
#     "tagatcaactcggccatacacaacaacttgacctacataacatgatacataacttatccaaccaatctggaagcaaacaa"
#     "ccaaaccaaattctgactaatctcatttaacttatagctacccactaaaacctcttgcccttaactcggatgaagacacc"
#     "aaggtctaaaaagaagagctcaacatcgataaggaaggagatggatcaaatccaacttcaagaggaatcaagatccacag"
#     "ttcaggatgaaagtctttccttattatactcttccttgccacagcccatacttgcaataaccatacctcatctgacataa"
#     "tagatggctatacatacaatatccatattttccaaatcatcaactaattatttttttttaaaaaaaacttgaaacaaaac"
#     "ttttgattcctaaaccaaatgagaaactaaatttcaaaccaatctagttacatcccttttattactaatctcgaggacga"
#     "gatttcttttaaggggggtaggacttgtaaaccctgaataatgctttaaggaattttagctagttctttatttcataaga"
#     "ataatggtaaggatctataggactttagttaatcaagaaaaaaatattaaagttatttttagtatgtgtctaagtgttga"
#     "agacatgttccttttgaatcaataggtgtgaattatttcaccataataatacctattggattatacaatgatgtgaattt"
#     "ctcgcttattgaatatcataaatgaaaataaatacatgatcaaattaagagaggaataaatacacataaaatattattgc"
#     "gtcatgctaggattattttgtttgtcaaaactcagaattgaatttgtttgaaaaaaattgaaattgaaacttgaaaataa"
#     "aatgaaataaagaaaaggaattgaaaaacagaaaataaaagaaatcgtcgtttgggccgaatccactacatcggcccacc"
#     "tcaccccacacctgcgtggcccgcgatccgtgcttgggtcaccgacacgtgggccctctgtgtcagcctcttcgctggct"
#     "gtgcgggttgttgtgggtggctgacttgtggagccgatttgtcagccgagtttttctcttccccaacaaacttccgccat"
#     "ccgtgctccatggtcagctcaacagaacctccataaatccatgcgccacaatggctggcaaactcaaacgccgataccgc"
#     "catgagttgttacatcctccctacccatcgcacgctgcagttgacgaggaactctgcctttgcggactataaaacggagc"
#     "tcgtacccccctttttcctggccatcttcacagagtctgcagatggaggaatcgggccgccattggggtattccacttcc"
#     "acccacgctgctggaacccgctcgaaggctctggggttcacgcaagtccactgaaaggcagacaaggcagaatctgcacg"
#     "cgaagatgacatgtgcttcatgaatccctcatcgacggagatccgccgccgtgggaccgctatggcgtgtggatggatac"
#     "tgccatcccaaatcaaggtatggtagccactatcgtgttcgactcctgttcagctgcgcgtagaacatagcggcatgggc"
#     "aaaagagcactggtacgcggaataatcgctcgccggtggtgctctgtcgcggcacggcgtgctgtcaagttggtgggcgc"
#     "cgtagggaagaagacgcgggtctgggagcttcgtcggttaccctggggcgtgttcgatgcggtgctcgtttccatccttc"
#     "tgtccatgctcgtgccgcctgggcactcgtgctcgccgtcgtcatgtcgacagggcgccgcctcctagtttctgtcatgg"
#     "ctctggtccgcacctcaccaggatccagccccaagctgccagcaacctccagcgttgctcatcgctcaccgcacactgcc"
#     "ttgcgcccccgcccagttctgctctcccttcgtcgctttgcggctctgccattgtttggacgtccggcctctggctttgg"
#     "cggccgccacccgccatctccaccttgctgccggttcgatctcagagaaggcactcagagcatcaccgctataccaggga"
#     "agttgtgttcaagactcaccacgcctggggcgttcttccgtggccggaatttctcgtcagtgcggggtttgtgccgtcgt"
#     "ccgccccgcaccatggtcatgggtgccgctgcaccaagcaccgttaacgattcccccatggcgctctccatcgaccgtac"
#     "catgtgtattcctctttgaattagctttcggacaccaattgtgggggggaaggtatcaacggcgtggctccaccgtggtc"
#     "ctgtgtgggcgccgccgtgctggagctgcgtcatcgctgtcctggtcgggattagggtgttgatcgggcaccgttcgatc"
#     "tcaatctaaggctctggattaaaacctagaatatcgtttcgggattttggtctcaatcgtccgatcagatcgtggggcta"
#     "cgattaaatctgcacccccctcgtccccatgttatggtggccgttgatctcaggatcggcggctcgggtgagttggctgg"
#     "tttggatctaatctcggcccccgatatcggatcgggcggccaggatgctcgtataccccttcggcctggcgatctttcta"
#     "aagagcccctcggcttgttccaaataaacccgcagtccacactgcgggagttctgagtcttggtgtcttttgcgcagaac"
#     "cccctgggtttctattaaatagtggccgcagtccaagtaactcagaaatttattaaatttaatacagaaattctttttaa"
#     "tgcaaaataatgtcccaaacttaagaaattcataacttggtgaatttagctccaaattgattcattccagtggcattaat"
#     "tttgtttaagtattatttattacctagtacctttgtttagccacaaaacttgaattaaaattgttcatttggattaaact"
#     "attctaagcactaaataattttggaaattcataacttaataaccgtagttccgaatttagcggttctcgaacccacgatc"
#     "tcgtaacgacgcgtagattattattatgtatcttgttcttatgtttggtgtgatgttaattttgcctataccatatatgt"
#     "ctgtattgctacgtttagcggggaggacgagtcacctgaagatcatcctggtacctggaatctcaagtcccaggcaagtt"
#     "gtgcccttgatcacttctttttacccactcatgttctaattaatcataatgatctgcataggttaattttgatgggaccc"
#     "aataggttaccctagtttgattatctttataccttgtttaccactgaactttttgggtagtacttgctagtgctttatgt"
#     "ggttttgggtatgaagatacattattcatgatcacacttttattatctgtttattatcactgttcatgataagatcatta"
#     "tgttaattggaacatggagcgaccacccgggaaaacagtgctaccacaagggtataatgggacgccctcggctgattaat"
#     "taggaaagctagtggaggactaccttacccgaaaggggcaagggcagtaggggagtggtcagtgtagggaggccctctgg"
#     "aggattttgctgcgatggcggtcctgcaagggattcctgcattggagcttcctataaactgtagcgggttttctgaagct"
#     "agtggaactttgtaaaggcctcgtagtgttaccctgcctcgcctcctcggtagaggtgtatgggaagtcgcgatcccttg"
#     "gcagatgggtaacatgacttgtgggtaaagatgcgcaacctctgcagagtgtaaaactggtatactagccgtgctcacgg"
#     "tcatgagcagctcggaccctcacatgattaatttatggaacttaaattcaatttgtcatatgcattgcatcgcaggtgat"
#     "gttgttacttctgttctactatttaattgggctggtatttacttatacttagtaattgctaataaaattttgaccaactt"
#     "taaaagcaatgctcagcttcaaccatcttctttggtaagccttacacttcacgtgagctcccgcctttggcgagttcatg"
#     "cacattattccccacaacttgttgagcgatgaacgtatgtgagctcactcttgctgtctcacacccccccacacaggtca"
#     "agaacaggtaccacaggatgaggcgcatgaaggatgctgtgacgagttcgtgagaggtctaggtcgtcgtctcccagtca"
#     "acttcgggttgctggaccgttgtctccatataatgtaattatttatttattttgtacagaactccgattatatagtaaag"
#     "atgtgacattcgatcctgtgccatgattcatcatatgtgtgagacttggtcccagcacacctggtgattatgttcgcgcc"
#     "cgggtcttggtgccccgaaacccgggtgtgaca"
# )
# helitron = sequence.upper()
# features, graph = dotplot(helitron, helitron, wsize=15, nmatch=12, scatter=True)