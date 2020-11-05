import sys
import click
from tqdm import tqdm


@click.command()
@click.option("--med1250_pmid", type=str, help="path to MED1250 PubMed IDs.")
@click.option("--biotext_pmid", type=str, help="path to BioText PubMed IDs.")
def main(med1250_pmid, biotext_pmid):
    eval_pmid = set()

    with open(med1250_pmid) as f:
        for line in f:
            eval_pmid.add(line.strip())

    with open(biotext_pmid) as f:
        for line in f:
            eval_pmid.add(line.strip())

    for line in tqdm(sys.stdin):
        line = line.strip()
        if line == "pmid":
            continue
        if line in eval_pmid:
            continue
        sys.stdout.write(line + "\n")

if __name__ == '__main__':
    main()
