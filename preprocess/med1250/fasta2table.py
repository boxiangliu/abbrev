import sys
sys.path.append(".")
from utils import fasta2table
from collections import defaultdict
import click


@click.command()
@click.option("--in_fn", type=str, help="Path to input file.")
@click.option("--out_fn", type=str, help="Path to output file.")
def main(in_fn, out_fn):
    with open(in_fn) as f:
        data = fasta2table(f, defaultdict(list))

    data.to_csv(out_fn, index=False, sep="\t")

if __name__ == "__main__":
    main()
