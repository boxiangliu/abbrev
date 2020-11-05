import sys
from collections import Counter
from tqdm import tqdm
import click

@click.command()
@click.option("--topk", type=int, default=5, help="Only output top K results.")
def main(topk):
    freq = Counter()
    sys.stderr.write("Calculating 3D frequency...\n")
    for line in tqdm(sys.stdin):

        if line.startswith("  "):
            split_line = line.strip().split("|")
            comment = split_line[3]
            if comment == "input": continue

            sf = split_line[0]
            lf = split_line[1]
            rank = int(split_line[3].replace("bqaf",""))
            if rank > topk: continue

            freq[(sf, lf, rank)] += 1

    sys.stderr.write("Writing to output...\n")
    for k, freq in tqdm(freq.items()):
        sf = k[0]
        lf = k[1]
        rank = k[2]
        sys.stdout.write(f"{sf}\t{lf}\t{rank}\t{freq}\n")

if __name__ == "__main__":
    main()