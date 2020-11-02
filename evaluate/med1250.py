import pandas as pd
from collections import defaultdict
import re
import sys
sys.path.append(".")
from utils import fasta2table, rerank2table
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

label_fn = "../processed_data/preprocess/med1250/fltr_answerable/MED1250_labeled"
ab3p_fn = "../processed_data/evaluate/MED1250/MED1250_ab3p"
bert_ab3p_ft_fn = "../processed_data/evaluate/MED1250/MED1250_bert-ab3p-ft"
bert_squad_ft_fn = "../processed_data/evaluate/MED1250/MED1250_bert-squad-ft"
bert_ab3p_ft_proposal_fn = "../processed_data/evaluate/MED1250/MED1250_bert-ab3p-ft_proposal"
bert_squad_ft_proposal_fn = "../processed_data/evaluate/MED1250/MED1250_bert-squad-ft_proposal"

# Reranked:
ken_dir = "/mnt/big/kwc/pubmed/Boxiang/rerank/"
bert_ab3p_ft_rerank_fn = f"{ken_dir}/MED1250_bert-ab3p-ft.rerank"
bert_squad_ft_rerank_fn = f"{ken_dir}/MED1250_bert-squad-ft.rerank"
bert_ab3p_ft_proposal_rerank_fn = f"{ken_dir}/MED1250_bert-ab3p-ft_proposal.rerank"
bert_squad_ft_proposal_rerank_fn = f"{ken_dir}/MED1250_bert-squad-ft_proposal.rerank"
bert_ab3p_ft_filtered_rerank_fn = f"{ken_dir}/MED1250_bert-ab3p-ft_filtered.rerank"
bert_squad_ft_filtered_rerank_fn = f"{ken_dir}/MED1250_bert-squad-ft_filtered.rerank"

out_dir = "../processed_data/evaluate/med1250/"
os.makedirs(out_dir, exist_ok=True)

methods = ["label", "ab3p", "ab3p_ft", "squad_ft", "ab3p_ft_proposal", "squad_ft_proposal",
           "ab3p_ft_rerank", "squad_ft_rerank", "ab3p_ft_proposal_rerank", "squad_ft_proposal_rerank"]
fns = [label_fn, ab3p_fn, bert_ab3p_ft_fn, bert_squad_ft_fn,
       bert_ab3p_ft_proposal_fn, bert_squad_ft_proposal_fn, bert_ab3p_ft_rerank_fn,
       bert_squad_ft_rerank_fn, bert_ab3p_ft_proposal_rerank_fn, bert_squad_ft_proposal_rerank_fn]
assert len(methods) == len(fns)

dfs = dict()
for method, fn in zip(methods, fns):
    with open(fn) as f:
        if "rerank" in method:
            df = rerank2table(f, defaultdict(list))
            df = df[df["comment"] != "ab3p"]
            df["comment"] = 1
        elif "ft" in method:
            df = fasta2table(f, defaultdict(list))
            df = df[df["comment"] != "ab3p"]
            df["comment"] = df["comment"].apply(
                lambda x: int(x.replace("bqaf", "")))
        elif method == "ab3p":
            df = fasta2table(f, defaultdict(list))
            df["comment"] = 1

        dfs[method] = df


########
# Help #
########
# Output: Recall
# Definition: ({sf-lf pair in ab3p} intersect {sf-lf pairs in label}) /
# {sf-lf pairs in label}
label = dfs["label"]
intxns = dict()
metrics = defaultdict(list)
for method in methods[1:]:
    for k in range(1, 6):
        df = dfs[method].loc[lambda x: x["comment"] <= k]
        intxn = pd.merge(label, df, how="inner", on=[
            "pmid", "type", "sent_no", "sent", "sf", "lf"], suffixes=("_label", "_pred"))
        intxn = intxn[["sf", "lf", "pmid", "type", "sent_no",
                       "sent", "score_label", "score_pred", "comment_label"]]
        intxn = intxn.groupby(["sf", "lf", "pmid", "type", "sent_no",
                               "sent", "score_label", "comment_label"]).agg(score_pred=("score_pred", "max")).reset_index()
        intxns[(method, k)] = intxn
        recall = intxn.shape[0] / label.shape[0]
        precision = intxn.shape[0] / df.shape[0]
        print(f"{method} recall @ {k}: {recall:.3f}")
        print(f"{method} precision @ {k}: {precision:.3f}")
        metrics["method"].append(method)
        metrics["top"].append(k)
        metrics["recall"].append(recall)
        metrics["precision"].append(precision)


metrics = pd.DataFrame(metrics)

plt.close()
plt.ion()
sns.lineplot(data=metrics, x="recall", y="precision", hue="method", marker="o")
plt.show()
plt.savefig(f"{out_dir}/pr_curve.pdf")
# Which abbreviations aren't captured by Ab3P?
# merged[merged["score_ab3p"].isna()].groupby("comment").nunique()
# merged[merged["score_ab3p"].notna()].groupby("comment").nunique()
# Most of the entries that ab3p found has no comments.
# A lot of missed entries are due to "nch" and "ord"
