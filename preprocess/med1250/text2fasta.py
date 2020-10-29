import os
import re
import stanza
from tqdm import tqdm
import click
import sys
sys.path.append(".")
from utils import create_dir_by_fn

########
# Help #
########
# Input format:
# <PMID>
# Title
# Abstract
#   <SF>|<LF>
# \n
#
# Output format:
# ><PMID>|[ta]|<sentence no>
# Sentence
#   <SF>|<LF>


@click.command()
@click.option("--med1250_fn", type=str, help="Path to the MED1250 file.")
@click.option("--out_fn", type=str, help="Path to the output file.")
def main(med1250_fn, out_fn):

    create_dir_by_fn(out_fn)
    nlp = stanza.Pipeline("en", processors="tokenize", tokenize_batch_size=64)

    with open(med1250_fn, encoding="Windows-1252") as fin, \
            open(out_fn, "w") as fout:

        sf = "NOT_A_SHORT_FORM"
        lf = "NOT_A_LONG_FORM"
        for line in tqdm(fin):

            line = line.rstrip()
            if re.match("^[0-9]+$", line):  # PMID
                pmid = line
                next_line_is_title = True
                sflfs = []

            elif line.startswith("  "):  # long form and short form pairs
                split_line = line.lstrip().split("|")
                # list: [short form, long form, comment]
                sflfs.append([split_line[0], split_line[1], "none"])

            elif line == "":  # Article change
                for typ, sentences in [("t", title_sentences), ("a", abstract_sentences)]:
                    for i, sentence in enumerate(sentences):
                        sentence = sentence.text
                        fasta_header = f">{pmid}|{typ}|{i}\n"
                        fout.write(fasta_header)
                        fout.write(sentence + "\n")
                        for sf, lf, comment in sflfs:
                            if ((f"({sf})" in sentence) or (f"[{sf}]" in sentence)) \
                                    and (lf in sentence):
                                fout.write(f"  {sf}|{lf}|1|{comment}\n")
                sf = "NOT_A_SHORT_FORM"
                lf = "NOT_A_LONG_FORM"

            elif line.startswith("//"):  # Comment lines
                # If the line starts with //, discard the comment
                # If the line starts with //!, keep the string afterwards. Except for //!syn
                # If the line starts with //*, this indicate synonyms, discard the line
                # It is possible to have two comments for a single lf-sf pair

                if line.startswith("//!") and not line.startswith("//!syn"):
                    if sflfs[-1][2] == "none":
                        sflfs[-1][2] = line.strip().replace("//!", "")
                    else:
                        sflfs[-1][2] += "," + line.strip().replace("//!", "")
                else:
                    pass

            else:  # Title or abstract
                doc = nlp(line.strip())
                if next_line_is_title:
                    title_sentences = doc.sentences
                    next_line_is_title = False
                else:
                    abstract_sentences = doc.sentences


if __name__ == "__main__":
    main()
