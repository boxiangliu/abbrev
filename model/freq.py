import sys

freq_fn = "/mnt/scratch/boxiang/projects/abbrev/processed_data/preprocess/count_abbrev/count.tsv"
freq = dict()
with open(freq_fn) as f:
    for line in f:
        split_line = line.strip().split("\t")
        freq[split_line[0]] = split_line[1]

sys.stdout.write("sf\tfreq\n")

for i, line in enumerate(sys.stdin):
    if i == 1 and line.startswith("sf"):
        continue

    out_line = []
    sf = line.strip()
    out_line.append(sf)
    out_line.append(freq[sf] if sf in freq else "0")
    out_line = "\t".join(out_line)
    sys.stdout.write(out_line + "\n")
