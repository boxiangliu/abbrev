import sys
for line in sys.stdin:
    if line.startswith("text"):
        text = line.strip().split("\t")[1]
    elif line.startswith("id"):
        id_ = line.strip().split("\t")[1]
    elif line.startswith("annotation:\tSF"):
        sf = line.strip().split("\t")[2]
    elif line.startswith("annotation:\tLF"):
        lf = line.strip().split("\t")[2]
        sys.stdout.write("{}\t{}\t{}\t{}\n".format(lf, sf, text, id_))