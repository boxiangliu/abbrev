import sys
from tqdm import tqdm

sys.stdout.write(f"sf\tlf\tgood_sf\tgood_lf\n")
for line in tqdm(sys.stdin):
    if line.startswith("  "):
        split_line = line.strip().split("|")
        if split_line[3] == "input":
            if float(split_line[2]) == 1.0:
                valid_sf = 1
                gold_lf = split_line[1]
            else: 
                valid_sf = 0
                gold_lf = "none"
        else:
            sf = split_line[0]
            lf = split_line[1]
            if lf == gold_lf:
                correct_lf = 1
            out_line = f"{sf}\t{lf}\t{valid_sf}\t{correct_lf}\n"
            sys.stdout.write(out_line)




