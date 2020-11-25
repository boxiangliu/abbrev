import sys
from tqdm import tqdm

sys.stdout.write(f"sf\tlf\tgood_sf\tgood_lf\tgold_answer\n")
for line in tqdm(sys.stdin):
    if line.startswith("  "):
        split_line = line.strip().split("|")
        sf, lf = split_line[0], split_line[1]
        if split_line[3] == "input":
            valid_sf, gold_lf = (1, lf) \
                if float(split_line[2]) == 1.0 else (0, "none")
            correct_lf, gold_answer = 1, 1
        else:
            correct_lf = int(lf == gold_lf)
            gold_answer = 0
        out_line = f"{sf}\t{lf}\t{valid_sf}\t{correct_lf}\t{gold_answer}\n"
        sys.stdout.write(out_line)
