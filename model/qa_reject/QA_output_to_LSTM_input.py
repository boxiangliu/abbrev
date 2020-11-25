import sys
from tqdm import tqdm

sys.stdout.write(f"sf\tlf\tgood_sf\tgood_lf\n")
for line in tqdm(sys.stdin):
    if line.startswith("  "):
        split_line = line.strip().split("|")
        if split_line[3] == "input":
            valid_sf, gold_lf = 1, split_line[1] \
                if float(split_line[2]) == 1.0 else 0, "none"
        else:
            sf, lf = split_line[0], split_line[1]
            correct_lf = int(lf == gold_lf)
            out_line = f"{sf}\t{lf}\t{valid_sf}\t{correct_lf}\n"
            sys.stdout.write(out_line)
