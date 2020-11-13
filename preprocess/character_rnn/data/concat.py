import click
import sys

@click.command()
@click.option("--pos", type=str, help="Path to files with positive examples. Multiple files should be separated by commas.")
@click.option("--neg", type=str, help="Path to files with negative examples. Multiple files should be separated by commas.")
def main(pos, neg):
    pos = pos.split(",")
    neg = neg.split(",")
    for label, flist in {1: pos, 0: neg}.items():
        for fn in flist:
            with open(fn) as f:
                for line in f:
                    line = line.strip()
                    sys.stdout.write(f"{line}\t{label}\n")
