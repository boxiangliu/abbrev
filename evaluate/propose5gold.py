import sys
sys.path.append(".")
from utils import fasta2table
import pandas as pd

proposal_fn = "../processed_data/model/propose/MED1250_proposal"
gold_fn = "../processed_data/preprocess/med1250/fltr_answerable/MED1250_labeled"


proposal = fasta2table(proposal_fn)
gold = fasta2table(gold_fn)
union = pd.merge(proposal.drop(columns=["lf", "comment"]), gold,
                 how="outer", on=["pmid", "type", "sent_no", "sent", "sf"], suffixes=("_p", "_g"))


# What is missing in the proposals?
missing = union[union.score_p.isna()]
missing.shape
for i in range(missing.shape[0]):
    print(missing.iloc[i].sf)
    print(missing.iloc[i].lf)
    print(missing.iloc[i].sent)
    print()

extra = union[union.score_g.isna()]
extra.shape
with open("test","w") as f:
    for i in range(extra.shape[0]):
        f.write("  " + extra.iloc[i].sf + "\n")
        f.write(extra.iloc[i].sent + "\n")
        f.write("\n")


common = union[union.score_g.notna() & union.score_p.notna()]
common.shape
with open("test2","w") as f:
    for i in range(common.shape[0]):
        f.write("  " + common.iloc[i].sf + "\n")
        f.write(common.iloc[i].sent + "\n")
        f.write("\n")

proposal
