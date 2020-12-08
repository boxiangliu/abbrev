import sys

sys.stdout.write(f"sf\tlf\tfreq\n")
for line in sys.stdin:
    freq, text = line.strip().split("\t")
    lf, sf = text.rsplit(" (", 1)
    sys.stdout.write(f"{sf}\t{lf}\t{freq}\n")
