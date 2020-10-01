import re
import os

in_fn = "../processed_data/preprocess/sentence/pubmed19n0972.txt"
out_dir = "../processed_data/preprocess/abbrev/"
os.makedirs(out_dir, exist_ok=True)
out_fn = os.path.join(out_dir, "pubmed19n0972.txt")

with open(in_fn) as fin, open(out_fn, "w") as fout:
    for line in fin:
        split_line = line.strip().split("|")
        pmid = split_line[0]
        typ = split_line[1]
        line_no = split_line[2]
        text = split_line[3]

        for m in re.finditer("\((.*?)\)", text):
            m_text = m.group(1)
            start = m.start(1)
            end = m.end(1)
            out = f"{pmid}|{typ}|{line_no}|{start},{end}|{m_text}\n"
            fout.write(out)
