import sys
sys.path.append(".")
from utils import fasta2table
import pandas as pd
import os

proposal_fn = "../processed_data/model/propose/MED1250_filtered"
gold_fn = "../processed_data/preprocess/med1250/fltr_answerable/MED1250_labeled"
out_dir = "../processed_data/evaluate/proposal5gold/"
os.makedirs(out_dir, exist_ok=True)


proposal = fasta2table(proposal_fn)
gold = fasta2table(gold_fn)
union = pd.merge(proposal.drop(columns=["lf", "comment"]), gold,
                 how="outer", on=["pmid", "type", "sent_no", "sent", "sf"], suffixes=("_p", "_g"))


# What is missing in the proposals?
missing = union[union.score_p.isna()]
with open(f"{out_dir}/missing", "w") as f:
    for i in range(missing.shape[0]):
        f.write("  " + missing.iloc[i].sf + "\n")
        f.write(missing.iloc[i].sent + "\n")
        f.write("\n")


extra = union[union.score_g.isna()]
with open(f"{out_dir}/extra","w") as f:
    for i in range(extra.shape[0]):
        f.write("  " + extra.iloc[i].sf + "\n")
        f.write(extra.iloc[i].sent + "\n")
        f.write("\n")


common = union[union.score_g.notna() & union.score_p.notna()]
with open(f"{out_dir}/common","w") as f:
    for i in range(common.shape[0]):
        f.write("  " + common.iloc[i].sf + "\n")
        f.write(common.iloc[i].sent + "\n")
        f.write("\n")


with open(f"{out_dir}/statistic", "w") as f:
    f.write(f"gold: {gold.shape[0]}\n")
    f.write(f"proposal: {proposal.shape[0]}\n")
    f.write(f"proposal - gold: {extra.shape[0]}\n")
    f.write(f"gold - proposal: {missing.shape[0]}\n")
    f.write(f"common: {common.shape[0]}\n")

