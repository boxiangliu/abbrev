import sys
import click

@click.command()
@click.option("--exclude", type=str, help="Files containing sf-lf pairs to exclude. Multiple files can be separated by commas.")
def main(exclude):
    exclude_pairs = set()
    for fn in exclude.split(","):
        with open(fn) as f:
            for line in f:
                exclude_pairs.add(line.strip())
    sys.stderr.write(f"{len(exclude_pairs)} SF-LF pairs in the excluded list.")

    n = 0
    for line in sys.stdin:
        if line.startswith("sf"): continue
        split_line = line.strip().split("\t")
        score = float(split_line[2])
        freq = int(split_line[4])
        if (freq > 1) and (score + 0.005 * freq) >= 1:
            out_line = "\t".join(split_line[:2])
            if out_line in exclude_pairs: 
                sys.stderr.write(f"Excluding {out_line}\n")
                n += 1
                continue
            sys.stdout.write(out_line + "\n")

    sys.stderr.write(f"Excluded {n} pairs.")

if __name__ == '__main__':
    main()