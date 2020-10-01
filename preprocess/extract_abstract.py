import xml.etree.ElementTree as ET
import gzip
import os
import click

def get_title(PubmedArticle):
    title = PubmedArticle[0][2].find("ArticleTitle")
    if title == None:
        return None

    assert title.tag == "ArticleTitle"
    return title.text


def get_abstract(PubmedArticle):
    abstract = PubmedArticle[0][2].find("Abstract")
    if abstract == None:
        return None
    else:
        abstract_text = abstract[0]

    assert abstract_text.tag == "AbstractText"
    return abstract_text.text


def get_pmid(PubmedArticle):
    pmid = PubmedArticle[0][0]
    assert pmid.tag == "PMID"
    return pmid.text


@click.command()
@click.option("--in_fn", type=str, help="Input file.")
@click.option("--out_fn", type=str, help="Output file.")
def main(in_fn, out_fn):
    with gzip.open(in_fn, "r") as in_handle:
        xml = ET.parse(in_handle)
        root = xml.getroot()

    with open(out_fn, "w") as fout:
        for PubmedArticle in root:
            title = get_title(PubmedArticle)
            abstract = get_abstract(PubmedArticle)
            pmid = get_pmid(PubmedArticle)

            if title != None:
                out = f"{pmid}|t|{title}\n"
                fout.write(out)

            if title != None:
                out = f"{pmid}|a|{abstract}\n"
                fout.write(out)


if __name__ == "__main__":
    main()