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


def main():

    nlp = stanza.Pipeline(lang="en", processors="tokenize")
    proposals, sfs, lfs = [], [], []
    for line in tqdm(sys.stdin):

        if line.startswith("text:"):

            if (sfs != []) or (proposals != []):
                for sentence in text.sentences:
                    for sf, lf in zip(sfs, lfs):
                        if (sf in sentence.text) and (lf in sentence.text):
                            out_line = f"sentence:\t{sentence.text}\n"
                            out_line += f"SF:\t{sf}\n"
                            out_line += f"question:\tWhat does {sf} stand for?\n"
                            out_line += f"answer:\t{lf}\n\n"
                            sys.stdout.write(out_line)
                    for proposal in bad_proposals:
                        if "(" + proposal in sentence.text:
                            out_line = f"sentence:\t{sentence.text}\n"
                            out_line += f"SF:\t{proposal}\n"
                            out_line += f"question:\tWhat does {proposal} stand for?\n"
                            out_line += f"answer:\tN/A\n\n"
                            sys.stdout.write(out_line)
                proposals, sfs, lfs = [], [], []

            text = line.strip().split("\t")[1]
            find(text, proposals)
            bad_proposals = [x for x in proposals if x not in sfs]
            text = nlp(text)

        elif line.startswith("annotation:"):
            split_line = line.strip().split("\t")
            if split_line[1].startswith("SF"):
                sfs.append(split_line[2])
            elif split_line[1].startswith("LF"):
                lfs.append(split_line[2])


if __name__ == "__main__":
    main()
