import sys

sfs = []
lfs = []
for line in sys.stdin:
    if line.startswith("annotation:"):
        split_line = line.strip().split("\t")
        if split_line[1].startswith("SF"):
            sfs.append(split_line[2])
        else:
            lfs.append(split_line[2])

assert len(lfs) == len(sfs)

for sf, lf in zip(sfs, lfs):
    sys.stdout.write(f"{sf}\t{lf}\n")