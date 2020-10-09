import os
import glob
from collections import defaultdict
import pandas as pd
import re

in_dir = "../processed_data/preprocess/ab3p/run_ab3p/"
out_dir = "../processed_data/preprocess/ab3p/summarize_ab3p/"
os.makedirs(out_dir, exist_ok=True)

container = defaultdict(list)
for fn in glob.glob(f"{in_dir}/pubmed*.out"):
    print(f"INPUT\t{fn}")
    with open(fn, encoding="ISO-8859-1") as fin:
        for line in fin:
            try:
                if re.match(">[0-9]+\|[at]\|[0-9]+"):
                    split_line = line.strip().replace(">","").split("|")
                    pmid = int(split_line[0])
                    typ = split_line[1]
                    sent = int(split_line[2])

                if line.startswith("  "):
                    split_line = line.strip().split("|")
                    container["sf"].append(split_line[0])
                    container["lf"].append(split_line[1])
                    container["score"].append(float(split_line[2]))
                    container["pmid"].append(pmid)
                    container["type"].append(typ)
                    container["sent"].append(sent)
            except:
                print(line)

df = pd.DataFrame(container)
