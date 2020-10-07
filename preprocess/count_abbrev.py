import glob
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("TkAgg")
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
            abb_type = split_line[4]
            if abb_type == "0":
                abbrev = split_line[5]
                counter[abbrev] += 1


counter_filt = {x: count for x, count in counter.items() if count > 1}
counter_sort = sorted(counter.items(), key=lambda pair: pair[1], reverse=True)


with open(f"{out_dir}/count.tsv", "w") as fout:
    for x, count in counter_sort:
        fout.write(f"{x}\t{count}\n")


with open(f"{out_dir}/count.tsv", "r") as fin:
    counter_sort = list()
    for line in fin:
        split_line = line.strip().split("\t")
        abbrev = split_line[0]
        count = int(split_line[1])
        counter_sort.append((abbrev, count))


zipf = defaultdict(list)
for i, (x, count) in enumerate(counter_sort):
    zipf["rank"].append(i+1)
    zipf["freq"].append(count)


plt.ion()
plt.show()
plt.xscale("log")
plt.yscale("log")
plt.plot(zipf["rank"], zipf["freq"])
plt.xlabel("log(rank)")
plt.ylabel("log(frequency)")
plt.tight_layout()
plt.savefig(f"{out_dir}/rank_freq.pdf")


total_count = 0
cum_count = [0]
for x, count in counter_sort:
    total_count += count
    cum_count.append(cum_count[-1]+count)


cutoff = 0.8
for i, count in enumerate(cum_count):
    if count > total_count * cutoff:
        print(f"Top {i} words account for {cutoff*100}% of all occurances.")
        break

# Top 42278 words account for 80.0% of all occurances.