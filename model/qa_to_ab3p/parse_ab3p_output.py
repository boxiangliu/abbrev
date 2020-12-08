import sys

prev_state = None
sys.stdout.write("sf\tlf\tab3p\n")

for line in sys.stdin:
    if line.startswith(">"):
        if prev_state == "ab3p":
            continue
        elif prev_state == "squad":
            sys.stdout.write(f"{sf}\t{lf}\t0\n")

        pmid, type_, sent_no = line.replace(">", "").strip().split("|")
        prev_state = "header"

    elif line.startswith("squad:"):
        if prev_state == "ab3p":
            continue
        elif prev_state == "squad":
            sys.stdout.write(f"{sf}\t{lf}\t0\n")

        lf, sf = line.strip().split("\t")[1:3]
        prev_state = "squad"

    elif line.startswith("  "):
        ab3p_sf, ab3p_lf, score = line.strip().split("|")
        if (ab3p_sf == sf) and (ab3p_lf == lf):
            sys.stdout.write(f"{sf}\t{lf}\t{score}\n")
        
        prev_state = "ab3p"