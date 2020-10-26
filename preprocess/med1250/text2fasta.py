import os
import re
import stanza
from tqdm import tqdm
med1250_fn = "../data/MED1250/MED1250_labeled"
out_dir = "../processed_data/preprocess/med1250/text2fasta/"
os.makedirs(out_dir, exist_ok=True)

# Input format:
# <PMID>
# Title
# Abstract
#   <SF>|<LF>
# \n

# Output format:
# ><PMID>|[ta]|<sentence no>
# Sentence
#   <SF>|<LF>


def main(med1250_fn, out_dir, nlp):

    with open(med1250_fn, encoding="Windows-1252") as fin, \
            open(f"{out_dir}/MED1250_labeled", "w") as fout:

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
                for typ, sentences in [("t",title_sentences), ("a", abstract_sentences)]:
                    for i, sentence in enumerate(sentences):
                        sentence = sentence.text
                        fasta_header = f">{pmid}|{typ}|{i}\n"
                        fout.write(fasta_header)
                        fout.write(sentence + "\n")
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


nlp = stanza.Pipeline("en", processors="tokenize", tokenize_batch_size=64)
main(med1250_fn, out_dir, nlp)
