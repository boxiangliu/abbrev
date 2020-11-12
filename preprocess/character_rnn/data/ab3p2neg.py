import sys
import random

sf = []
lf = []
for line in sys.stdin:
    if line.startswith("sf"): continue
    split_line = line.strip().split("\t")
    sf.append(split_line[0])
    lf.append(split_line[1])


random.seed(42)
len_ = len(sf)
shuf_idx1 = random.sample(range(len_), len_)
shuf_idx2 = random.sample(range(len_), len_)
for i, j in zip(shuf_idx1, shuf_idx2):
    if i != j:
        sys.stdout.write(f"{sf[i]}\t{lf[j]}\n")
