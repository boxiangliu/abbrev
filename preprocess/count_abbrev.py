import glob
from collections import Counter
import matplotlib.pyplot as plt
import pickle 
import os

in_dir="../processed_data/preprocess/abbrev/"
out_dir="../processed_data/preprocess/count_abbrev/"
os.makedirs(out_dir, exist_ok=True)


counter = Counter()
for fn in glob.glob(f"{in_dir}/pubmed*.txt"):
    print(f"INPUT\t{fn}")
    with open(fn) as fin:
        for line in fin:
            split_line = line.strip().split("|")
            abbrev = split_line[4]
            counter[abbrev] += 1


with open(f"{out_dir}/count.pkl", "wb") as fout:
    pickle.dump(counter, fout)

