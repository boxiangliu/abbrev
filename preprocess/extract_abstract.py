import xml.etree.ElementTree as ET
import gzip
import os
import click
import html
from bs4 import BeautifulSoup


def html2text(text, rm_cr=True):
    try:
        assert type(text) == str
        text = html.unescape(html.unescape(text))
        soup = BeautifulSoup(text, features="html.parser")
        text = soup.get_text()
        if rm_cr:
            text = text.replace("\r", "")
        return text
    except:
        print(text)

def get_title(PubmedArticle):
    title = PubmedArticle[0].find("Article").find("ArticleTitle")
    if title == None or title.text == None:
        return None

    assert title.tag == "ArticleTitle"
    return html2text(title.text)


def get_abstract(PubmedArticle):
    abstract = PubmedArticle[0].find("Article").find("Abstract")
    if abstract == None:
        return None
    else:
        abstract_text = abstract[0]

        if abstract_text == None or abstract_text.text == None:
            return None

    assert abstract_text.tag == "AbstractText"
    return html2text(abstract_text.text)


def get_pmid(PubmedArticle):
    pmid = PubmedArticle[0][0]
    assert pmid.tag == "PMID"
    return pmid.text


@click.command()
@click.option("--in_fn", type=str, help="Input file.")
@click.option("--out_fn", type=str, help="Output file.")
def main(in_fn, out_fn):
    print(f"INPUT\t{in_fn}")
    print(f"OUTPUT\t{out_fn}")

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

            if abstract != None:
                out = f"{pmid}|a|{abstract}\n"
                fout.write(out)


if __name__ == "__main__":
    main()
