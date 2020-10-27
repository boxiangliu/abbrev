import pandas as pd
from collections import defaultdict
import re

label_fn = "/mnt/scratch/boxiang/projects/abbrev/processed_data/preprocess/med1250/fltr_answeralbe/MED1250_labeled"

def fasta2table(f, container):
    for line in f:
        # import ipdb; ipdb.set_trace()
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
    return df

container = defaultdict(list)
with open(label_fn) as f:
    df = fasta2table(f, container)

