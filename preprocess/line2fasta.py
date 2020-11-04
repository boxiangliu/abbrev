import sys

for line in sys.stdin:
    split_line = line.split("|", 3)
    header = "|".join(split_line[:3])
    out_line = f">{header}\n{split_line[3]}"
    sys.stdout.write(out_line)
