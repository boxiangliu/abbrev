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
union[union.score_p.isna()]


proposal
