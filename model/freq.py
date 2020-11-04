import sys

freq_fn = "/mnt/scratch/boxiang/projects/abbrev/processed_data/preprocess/count_abbrev/count.tsv"
freq = dict()
with open(freq_fn) as f:
    for line in f:
        split_line = line.strip().split("\t")
        freq[line[0]] = int(line[1])

for line in sys.stdin:
    out_line = []
    sf = line.strip()
    out_line.append(sf)
    out_line.append(freq[sf])
    out_line = "\t".join(out_line)
    sys.stdout.write(out_line + "\n")
