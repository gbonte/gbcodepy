import re
import random

def countmotif(seq, mot):
    # Calculate length of the sequence and the motif.
    N = len(seq)
    M = len(mot)
    cnt = 0
    # Loop over each position in the sequence.
    for i in range(N):
        # Extract substring from position i to i+M.
        substring = seq[i:i+M]
        # If the motif is found in the substring (using regex search),
        # add 1 to the counter.
        if re.search(mot, substring):
            cnt += 1
    return cnt

seq = "ATGGTCAAATTAACTTCAATCGCCGCTGGTGTCGCTGCCATCGCTGCTACTGCTTCTGCAACCACCACTCTAGCTCAATCTGACGAAAGAGTCAACTTGGTGGAATTGGGTGTCTACGTCTCTGATATCAGAGCTCACTTAGCCCAATACTACATGTTCCAAGCCGCCCACCCAACTGAAACCTACCCAGTCGAAGTTGCTGAAGCCGTTTTCAACTACGGTGACTTCACCACCATGTTGACCGGTATTGCTCCAGACCAAGTGACCAGAATGATCACCGGTGTTCCAAGTGGTACTCCAGCAGATTAAAGCCAGCCATCTCCAGTGCTCTAAGTCCAAGGACGGTATCTACACTATCGCAAACTAAG"

R = 200
l = list(seq)
mot = "AAG"
count = countmotif(seq, mot)
print("Motif " + mot + " encountered " + str(count) + " times")
countperm = [0] * R
for r in range(R):
    permutl = l[:]
    random.shuffle(permutl)
    permuted_seq = "".join(permutl)
    countperm[r] = countmotif(permuted_seq, mot)
p = sum(1 for x in countperm if x >= count) / R
if p < 0.05:
    print("Motif " + mot + " significantly represented")
else:
    print("Motif " + mot + " NOT significantly represented")
