import sys
from tqdm import tqdm

sys.stdout.write(f"sf\tlf\tgood_sf\tgood_lf\tgold_answer\n")
for line in tqdm(sys.stdin):
    if line.startswith(">"):
        pmid, type_, sent_no = line.replace(">","").strip().split("|")
    if line.startswith("  "):
        split_line = line.strip().split("|")
        sf, lf, score, comment = split_line[0:4]
        score = float(score)
        if comment == "input":
            valid_sf, gold_lf, correct_lf = (1, lf, 1) \
                if score == 1.0 else (0, "none", 0)
            gold_answer = 1
        else:
            correct_lf = int(lf == gold_lf)
            gold_answer = 0
        out_line = f"{sf}\t{lf}\t{valid_sf}\t{correct_lf}\t{gold_answer}\t{pmid}\t{type_}\t{sent_no}\n"
        sys.stdout.write(out_line)
