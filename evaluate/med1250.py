import pandas as pd
from collections import defaultdict
import re
import sys
sys.path.append(".")
from utils import fasta2table


label_fn = "../processed_data/preprocess/med1250/fltr_answerable/MED1250_labeled"
ab3p_fn = "../processed_data/evaluate/MED1250/MED1250_ab3p"
bert_ab3p_ft_fn = "../processed_data/evaluate/MED1250/MED1250_bert_ab3p_ft"
bert_squad_ft_fn = "    "

methods = ["label", "ab3p", "ab3p_ft", "squad_ft"]
fns = [label_fn, ab3p_fn, bert_ab3p_ft_fn, bert_squad_ft_fn]
dfs = dict()
for method, fn in zip(methods, fns):
    with open(fn) as f:
        df = fasta2table(f, defaultdict(list))
        if method.endswith("ft"):
            df = df[df["comment"] != "ab3p"].drop(columns="comment")
        dfs[method] = df

########
# Help #
########
# Output: Recall
# Definition: ({sf-lf pair in ab3p} intersect {sf-lf pairs in label}) /
# {sf-lf pairs in label}
label = dfs["label"]
intxns = dict()
for method in methods[1:]:
    intxn = pd.merge(label, dfs[method] , how="inner", on=[
                      "pmid", "type", "sent_no", "sent", "sf", "lf"], suffixes=("_label", "_pred"))
    intxn = intxn[["sf", "lf", "pmid", "type", "sent_no",
                     "sent", "score_label", "score_pred", "comment"]]
    intxn = intxn.groupby(["sf", "lf", "pmid", "type", "sent_no",
                     "sent", "score_label", "comment"]).agg(score_pred=("score_pred", "max")).reset_index()
    intxns[method] = intxn
    recall = intxn.shape[0] / label.shape[0]
    print(f"{method} recall: {recall:.3f}")

# pd.merge(label, intxns["ab3p_ft"], indicator=True, how="outer").loc[lambda x: x["_merge"] == "left_only"]
# pd.merge(label, intxns["squad_ft"], indicator=True, how="outer").loc[lambda x: x["_merge"] == "left_only"]
# pd.merge(label, intxns["ab3p"], indicator=True, how="outer").loc[lambda x: x["_merge"] == "left_only"]

# Which abbreviations aren't captured by Ab3P?
# merged[merged["score_ab3p"].isna()].groupby("comment").nunique()
# merged[merged["score_ab3p"].notna()].groupby("comment").nunique()
# Most of the entries that ab3p found has no comments.
# A lot of missed entries are due to "nch" and "ord"
