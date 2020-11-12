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
    with open(fn, encoding="utf-8") as fin:
        for line in fin:
            try:
                if re.match(">[0-9]+\|[at]\|[0-9]+", line):
                    split_line = line.strip().replace(">", "").split("|")
                    pmid = int(split_line[0])
                    typ = split_line[1]
                    sent_no = int(split_line[2])

                elif line.startswith("  "):
                    split_line = line.strip().split("|")
                    container["sf"].append(split_line[0])
                    container["lf"].append(split_line[1])
                    container["score"].append(float(split_line[2]))
                    container["pmid"].append(pmid)
                    container["type"].append(typ)
                    container["sent_no"].append(sent_no)
                    container["sent"].append(sent)

                else:
                    sent = line.strip()

            except:
                print("error")
                print(line)

df = pd.DataFrame(container)
df.to_csv(f"{out_dir}/ab3p_res.csv", sep="\t", index=False)


df2 = df.groupby(["sf", "lf", "score"]).agg(freq=("score", "count"), pmids=("pmid", "nunique")).sort_values("freq", ascending=False).reset_index()
df2.to_csv(f"{out_dir}/ab3p_freq.csv", sep="\t", index=False)


