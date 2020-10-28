import pandas as pd
from collections import defaultdict
import re
import sys
sys.path.append(".")
from utils import fasta2table


label_fn = "../processed_data/preprocess/med1250/fltr_answeralbe/MED1250_labeled"
ab3p_fn = "../processed_data/evaluate/MED1250/MED1250_ab3p"
bert_ab3p_ft_fn = "../processed_data/evaluate/MED1250/MED1250_bert_ab3p_ft"
bert_squad_ft_fn = "../processed_data/evaluate/MED1250/MED1250_bert_squad_ft"

container = dict()
for variable, fn in [("label", label_fn), ("ab3p", ab3p_fn),
                     ("bert_ab3p_ft", bert_ab3p_ft_fn), ("bert_squad_ft", bert_squad_ft_fn)]:
    with open(fn) as f:
        container[variable] = fasta2table(f, defaultdict(list))

########
# Help #
########
# Output: Recall
# Definition: ({sf-lf pair in ab3p} intersect {sf-lf pairs in label}) /
# {sf-lf pairs in label}
merged = pd.merge(label, ab3p, how="outer", on=[
                  "pmid", "type", "sent_no", "sent", "sf", "lf"], suffixes=("_label", "_ab3p"))
merged = merged[["sf", "lf", "pmid", "type", "sent_no",
                 "sent", "score_label", "score_ab3p", "comment"]]
intxn = merged[(merged["score_label"].notna()) &
               (merged["score_ab3p"].notna())]
recall = intxn.shape[0] / label.shape[0]
print(f"Recall: {recall:.3f}")

# Which abbreviations aren't captured by Ab3P?
merged[merged["score_ab3p"].isna()].groupby("comment").nunique()
merged[merged["score_ab3p"].notna()].groupby("comment").nunique()
# Most of the entries that ab3p found has no comments.
# A lot of missed entries are due to "nch" and "ord"
