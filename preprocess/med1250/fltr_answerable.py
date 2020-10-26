from tqdm import tqdm
import sys
sys.path.append(".")
from utils import create_dir_by_fn

med1250_fn = "../processed_data/preprocess/med1250/text2fasta/MED1250_labeled"
out_fn = "../processed_data/preprocess/med1250/fltr_answeralbe/MED1250_labeled"

create_dir_by_fn(out_fn)

with open(med1250_fn) as fin, open(out_fn, "w") as fout:
    for line in tqdm(med1250_fn):
        if line.startswith(">"):
            header = line
        elif line.startswith("  "):
            answer = line
            fout.write(header)
            fout.write(sentence)
            fout.write(answer)
        else:
            sentence = line