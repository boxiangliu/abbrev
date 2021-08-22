import sys
for line in sys.stdin:
    if line.startswith("text"):
        text = line.strip().split("\t")[1]
        print(text)
    elif line.startswith("annotation\tSF"):
        sf = line.strip().split("\t")[2]
        print(sf)
    elif line.startswith("annotation\tLF"):
        lf = line.strip().split("\t")[2]
        print(lf)
        sys.stdout.write("{}\t{}\t{}\n".format(lf, sf, text))

