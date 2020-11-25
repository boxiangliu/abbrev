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
    proposals, sfs, lfs = [], [], []
    for line in tqdm(sys.stdin):

        if line.startswith("id:"):
            pmid = int(line.strip().split("\t")[1])

        elif line.startswith("text:"):

            bad_proposals = [x for x in proposals if x not in sfs]

            if (sfs != []) or (proposals != []):
                for sent_no, sentence in enumerate(text.sentences):
                    for sf, lf in zip(sfs, lfs):
                        if (sf in sentence.text) and (lf in sentence.text):
                            out_line = "\t".join([sf, lf, "-1", "todo", pmid, text_type, sent_no, sentence.text])
                            # out_line = f"sentence:\t{sentence.text}\n"
                            # out_line += f"SF:\t{sf}\n"
                            # out_line += f"question:\tWhat does {sf} stand for?\n"
                            # out_line += f"answer:\t{lf}\n\n"
                            sys.stdout.write(out_line)
                    for proposal in bad_proposals:
                        if "(" + proposal in sentence.text:
                            out_line = "\t".join([sf, "none", "-1", "todo", pmid, text_type, sent_no, sentence.text])
                            # out_line = f"sentence:\t{sentence.text}\n"
                            # out_line += f"SF:\t{proposal}\n"
                            # out_line += f"question:\tWhat does {proposal} stand for?\n"
                            # out_line += f"answer:\tN/A\n\n"
                            sys.stdout.write(out_line)
                proposals, sfs, lfs = [], [], []

            text = line.strip().split("\t")[1]
            find(text, proposals)
            text = nlp(text)

            text_type, text_counter = get_text_type(text_counter)

        elif line.startswith("annotation:"):
            split_line = line.strip().split("\t")
            if split_line[1].startswith("SF"):
                sfs.append(split_line[2])
            elif split_line[1].startswith("LF"):
                lfs.append(split_line[2])


if __name__ == "__main__":
    main()
