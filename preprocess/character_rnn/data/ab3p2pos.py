import sys
for line in sys.stdin:
    if line.startswith("sf"): continue
    split_line = line.strip().split("\t")
    score = float(split_line[2])
    freq = int(split_line[4])
    if (freq > 1) and (score + 0.005 * freq) >= 1:
        out_line = "\t".join(split_line[:2])
        sys.stdout.write(out_line + "\n")