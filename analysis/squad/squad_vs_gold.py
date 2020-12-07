import sys
from tqdm import tqdm
import collections
from collections import OrderedDict
import click

@click.command()
@click.option("--proposal", type=str, help="Path to proposal file.")
@click.option("--squad", type=str, help="Path to SQuAD predictions.")
def main(proposal, squad):
    """
    container schema:
    pmid -> type_ -> sent_no -> sf -> gold_lf
                                  |-> squad_lf
    """
    container = OrderedDict()
    with open(squad) as f:
        for line in tqdm(f):
            if line.startswith("sf\tlf"):
                continue 
            else:
                sf, lf, gold_sf, gold_lf, gold_answer, \
                    pmid, type_, sent_no = line.strip().split("\t")

                container = init_container(container, pmid, type_, sent_no, sf)

                if gold_answer == "1":
                    gold_lf = lf
                    container[pmid][type_][sent_no][sf]["gold_lf"].append(gold_lf)
                elif gold_answer == "0":
                    squad_lf = lf
                    container[pmid][type_][sent_no][sf]["squad_lf"].append(squad_lf)

    with open(proposal) as f:
        for line in tqdm(f):
            sf, lf, score, comment, pmid, type_, sent_no, sent = line.strip().split("\t")
            if comment == "omit":
                sys.stderr.write("1\n")
                container = init_container(container, pmid, type_, sent_no, sf)
                container[pmid][type_][sent_no][sf]["gold_lf"].append(lf)
                container[pmid][type_][sent_no][sf]["squad_lf"].append(None)

    try:
        sys.stdout.write("pmid\ttype\tsent_no\tsf\tgold_lf\tsquad_lf\tcorrect_sf\tcorrect_lf\textra_pair\n")
        for pmid, pmid_content in container.items():
            for type_, type_content in pmid_content.items():
                for sent_no, sent_no_content in type_content.items():
                    for sf, sf_content in sent_no_content.items():
                        gold_lf = sf_content["gold_lf"][0]
                        squad_lfs = sf_content["squad_lf"]
                        for squad_lf in squad_lfs:
                            if squad_lf == gold_lf:
                                correct_sf, correct_lf, extra_pair = 1, 1, 0
                            elif gold_lf == "none":
                                correct_sf, correct_lf, extra_pair = 0, 0, 1
                            elif squad_lf != gold_lf:
                                correct_sf, correct_lf, extra_pair = 1, 0, 0
                            elif squad_lf is None:
                                sys.stderr.write("2\n")
                                correct_sf, correct_lf, extra_pair = 0, 0, 0
                            sys.stdout.write(f"{pmid}\t{type_}\t{sent_no}\t{sf}\t{gold_lf}\t{squad_lf}\t{correct_sf}\t{correct_lf}\t{extra_pair}\n")

    except BrokenPipeError:
        pass


def init_container(container, pmid, type_, sent_no, sf):
    if pmid not in container:
        container[pmid] = OrderedDict()
    if type_ not in container[pmid]:
        container[pmid][type_] = OrderedDict()
    if sent_no not in container[pmid][type_]:
        container[pmid][type_][sent_no] = OrderedDict()
    if sf not in container[pmid][type_][sent_no]:
        container[pmid][type_][sent_no][sf] = {"gold_lf":[], "squad_lf":[]}
    return container


if __name__ == '__main__':
    main()