import sys
import click
from tqdm import tqdm


@click.command()
@click.option("--med1250_PMID", type=str, help="path to MED1250 PubMed IDs.")
@click.option("--BioText_PMID", type=str, help="path to BioText PubMed IDs.")
def main(med1250_PMID, BioText_PMID):
    eval_PMID = set()

    with open(med1250_PMID) as f:
        for line in f:
            eval_PMID.add(line.strip())

    with open(BioText_PMID) as f:
        for line in f:
            eval_PMID.add(line.strip())

    for line in tqdm(sys.stdin):
        line = line.strip()
        if line == "pmid":
            continue
        if line in eval_PMID:
            continue
        sys.stdout.write(line + "\n")

if __name__ == '__main__':
    main()
