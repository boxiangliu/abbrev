import os
import re
import stanza
from tqdm import tqdm
med1250_fn = "../data/MED1250/MED1250_labeled"
out_dir = "../processed_data/preprocess/med1250/text2fasta/"


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
@click.option("--out_dir", type=str, help="Path to the output directory.")
@click.option("--write_labels/--no_labels", default=True, help="write_labels: output short form and long form pairs.")
def main(med1250_fn, out_dir, write_labels):

    os.makedirs(out_dir, exist_ok=True)
    nlp = stanza.Pipeline("en", processors="tokenize", tokenize_batch_size=64)

    if write_labels:
        suffix = "labeled"
    else:
        suffix = "unlabeled"

    with open(med1250_fn, encoding="Windows-1252") as fin, \
            open(f"{out_dir}/MED1250_{suffix}", "w") as fout:

        sf = "NOT_A_SHORT_FORM"
        lf = "NOT_A_LONG_FORM"
        for line in tqdm(fin):

            line = line.rstrip()
            if re.match("^[0-9]+$", line):  # PMID
                pmid = line
                next_line_is_title = True

            elif line.startswith("  "):  # long form and short form pairs
                split_line = line.lstrip().split("|")
                sf = split_line[0]
                lf = split_line[1]

            elif line == "":  # Article change
                for typ, sentences in [("t", title_sentences), ("a", abstract_sentences)]:
                    for i, sentence in enumerate(sentences):
                        sentence = sentence.text
                        fasta_header = f">{pmid}|{typ}|{i}\n"
                        fout.write(fasta_header)
                        fout.write(sentence + "\n")
                        if write_labels:
                            if (sf in sentence) and (lf in sentence):
                                fout.write(f"  {sf}|{lf}|1\n")
                sf = "NOT_A_SHORT_FORM"
                lf = "NOT_A_LONG_FORM"

            else:  # Title or abstract
                doc = nlp(line.strip())
                if next_line_is_title:
                    title_sentences = doc.sentences
                    next_line_is_title = False
                else:
                    abstract_sentences = doc.sentences


if __name__ == "__main__":
    main()
