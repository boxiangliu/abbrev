import sys
from collections import Counter
from tqdm import tqdm

freq = Counter()
for line in tqdm(sys.stdin):

    if line.starstwith("  "):
        split_line = line.strip().split("|")
        comment = split_line[3]
        if comment == "input":
            continue
        else:
            sf = split_line[0]
            lf = split_line[1]
            rank = split_line[3].replace("bqaf","")
            freq[(sf, lf, rank)] += 1

for k, freq in freq.items():
    sf = k[0]
    lf = k[1]
    rank = k[2]
    sys.stdout.write(f"{sf}\t{lf}\t{rank}\t{freq}\n")
