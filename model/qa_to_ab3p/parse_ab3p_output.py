import sys

prev_state = None
sys.stdout.write("sf\tlf\tab3p\tpmid\ttype\tsent_no\n")

for line in sys.stdin:
    if line.startswith(">"):
        if prev_state == "squad":
            sys.stdout.write(f"{sf}\t{lf}\t0\t{pmid}\t{type_}\t{sent_no}\n")

        pmid, type_, sent_no = line.replace(">", "").strip().split("|")
        prev_state = "header"

    elif line.startswith("squad:"):
        if prev_state == "squad":
            sys.stdout.write(f"{sf}\t{lf}\t0\t{pmid}\t{type_}\t{sent_no}\n")

        lf, sf = line.strip().split("\t")[1:3]
        sf = sf[1:-1]

        prev_state = "squad"

    elif line.startswith("  "):
        ab3p_sf, ab3p_lf, score = line.strip().split("|")
        if (ab3p_sf == sf) and (ab3p_lf == lf):
            sys.stdout.write(f"{sf}\t{lf}\t{score}\t{pmid}\t{type_}\t{sent_no}\n")
        else:
            sys.stdout.write(f"{sf}\t{lf}\t0\n")

        prev_state = "ab3p"