import sys

for line in sys.stdin:
    if line.startswith(">"):
        sys.stdout.write(line)

    elif line.startswith("  "):
        split_line = line.strip().split("|")
        if split_line[3] == "input":
            continue
        sf, lf, score = split_line[:3]
        sys.stdout.write(f"squad:\t{lf}\t({sf})\n")