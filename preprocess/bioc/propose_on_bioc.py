#!/usr/bin/env python
# Propose short forms from input text
import sys
import regex as re


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
    proposals = []
    gold_sfs = set()

    for line in sys.stdin:
        if line.startswith("text:"):  # header line
            line = line.strip().split("\t")[1]
            find(line, proposals)

        elif line.startswith("annotation:\tSF"):
            gold_sf = line.strip().split("\t")[2]
            gold_sfs.append(gold_sf)

    for proposal in proposals:
        if proposal in gold_sfs:
            sys.stdout.write(f"{proposal}\t1\n")
        else:
            sys.stdout.write(f"{proposal}\t0\n")


if __name__ == "__main__":
    main()
