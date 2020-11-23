import sys
import regex as re
import random


def propose_sf(text, container=[]):
    # The regex matches nested parenthesis and brackets
    matches = re.finditer(
        " ([\(\[](?>[^\(\)\[\]]+|(?1))*[\)\]])", text)

    for match in matches:
        start = match.start(1)
        end = match.end(1)
        match = match.group(1)

        # Remove enclosing parenthesis
        sf = match[1:-1]

        # recurse to find abbreviation within a match:
        container = propose_sf(sf, container=container)

        # extract abbreviation before "," or ";"
        sf = re.split("[,;] ", sf)[0]

        # Add to container:
        container.append(
            {"sf": sf, "match": match, "start": start, "end": end})

    return container


def propose_lf(text, sfs, seed=42):

    random.seed(seed)
    lfs = []
    for sf in sfs:
        segment = text[:sf["start"]].strip().split()
        lf_len = random.randint(1, 10)
        lf = " ".join(segment[-lf_len:])
        lfs.append(lf)

    return lfs


def combine_sf_lf(sfs, lfs, exclude=set()):
    pairs = []
    for sf, lf in zip(sfs, lfs):
        pair = (sf["sf"], lf)
        if pair[0] not in exclude:
            pairs.append(pair)
    return pairs


def propose_sf_lf(text, exclude):
    sfs = propose_sf(text, container=[])
    lfs = propose_lf(text, sfs)
    pairs = combine_sf_lf(sfs, lfs, exclude=exclude)
    return pairs


text = ""
gold = set()
for line in sys.stdin:
    if line.startswith("text:"):
        # Take care of last chunk
        pairs = propose_sf_lf(text, gold)
        for sf, lf in pairs:
            sys.stdout.write(f"{sf}\t{lf}\n")

        # Start the next chunk
        text = line.strip()
        gold = set()

    elif line.startswith("annotation:"):
        split_line = line.strip().split("\t")
        if split_line[1].startswith("SF"):
            sf = split_line[2]
        else:
            lf = split_line[2]
            gold.add(sf)

