import sys

for line in sys.stdin:
    split_line = line.split("|", 3)
    out_line = f">{split_line[:3]}\n{split_line[3]}"
    sys.stdout.write(out_line)
