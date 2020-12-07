#!/usr/bin/env python
# Propose short forms from input text
import sys
import regex as re
import stanza
from tqdm import tqdm

def find(text, container):
    # The regex matches nested parenthesis and brackets
    matches = re.findall(
        " ([\(\[](?>[^\(\)\[\]]+|(?1))*[\)\]])", text)
    for match in matches:
        # Remove enclosing parenthesis
        match = match[1:-1]

        # recurse to find abbreviation within a match:
        find(match, container)

        # extract abbreviation before "," or ";"
        match = re.split("[,;] ", match)[0]
        container.append(match)


def get_text_type(text_counter):

    text_counter += 1
    if text_counter == 1:
        text_type = "title"

    elif text_counter == 2:
        text_type = "abstract"
        text_counter = 0

    return text_type, text_counter


def main():

    nlp = stanza.Pipeline(lang="en", processors="tokenize")
    text_counter = 0
    proposals, sf_lf = [], {}
    sys.stdout.write("sf\tlf\tscore\tcomment\tpmid\ttype\tsent_no\tsent\n")

    for line in tqdm(sys.stdin):

        if line.startswith("id:"):
            pmid = line.strip().split("\t")[1]

        elif line.startswith("text:"):

            bad_proposals = [sf for sf in proposals if sf not in sf_lf]
            good_proposals = [(sf, sf_lf[sf]) for sf in proposals if sf in sf_lf]
            omitted = [(sf, sf_lf[sf]) for sf in sf_lf.keys() if sf not in proposals]
            if (good_proposals != []) or (bad_proposals != []) or (omitted != []):
                for sent_no, sentence in enumerate(text.sentences):
                    for sf, lf in good_proposals:
                        if (sf in sentence.text) and (lf in sentence.text):
                            out_line = "\t".join([sf, lf, "1", "todo", pmid, text_type, str(sent_no), sentence.text])
                            sys.stdout.write(out_line + "\n")
                    for proposal in bad_proposals:
                        if "(" + proposal in sentence.text:
                            out_line = "\t".join([proposal, "none", "-1", "todo", pmid, text_type, str(sent_no), sentence.text])
                            sys.stdout.write(out_line + "\n")
                    for sf, lf in omitted:
                        if (sf in sentence.text) and (lf in sentence.text):
                            out_line = "\t".join([sf, lf, "1", "omit", pmid, text_type, str(sent_no), sentence.text])
                            sys.stdout.write(out_line + "\n")

                proposals, sfs, lfs = [], [], []

            text = line.strip().split("\t")[1]
            find(text, proposals)
            text = nlp(text)

            text_type, text_counter = get_text_type(text_counter)

        elif line.startswith("annotation:"):
            split_line = line.strip().split("\t")
            if split_line[1].startswith("SF"):
                sf = split_line[2]
            elif split_line[1].startswith("LF"):
                lf = split_line[2]
                sf_lf[sf] = lf


if __name__ == "__main__":
    main()
