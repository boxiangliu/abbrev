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

import numpy as np

def logit2prob(logit):
    odds = np.exp(logit)
    prob = odds / (1 + odds)
    return prob


def pr(scores, labels):
    # The input to precision recall curve is usually a data frame with a few columns 
    # For example:
    # score    label
    # 0.1        0
    # 0.42       0
    # 0.53       0
    # 0.65       1
    # 0.88       0
    # 0.95       1
    # precision = No. correct predictions / No. total predictions
    # recall    = No. correct predictions / No. truths
    # Algorithm: 
    # Sort the score in decreasing value.
    # n_truths = sum(label)
    # for each row:
    #   n_total_predictions += 1
    #   n_correct_predictions += 1 if label == 1 else 0
    #   precision = n_correct_predictions / n_total_predictions
    #   recall = n_correct_predictions / n_truths
    assert len(scores) == len(labels)
    pairs = [(s, l) for s, l in sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)]
    n_truths = sum(labels)
    precisions = []
    recalls = []
    scores = []
    n_total_predictions = 0
    n_correct_predictions = 0
    for score, label in pairs:
        if score >= 0:
            n_total_predictions += 1
            n_correct_predictions += 1 if label == 1 else 0
            precisions.append(n_correct_predictions / n_total_predictions)
            recalls.append(n_correct_predictions / n_truths)
            scores.append(score)

    return precisions, recalls, scores



methods = ["label", "ab3p", "ab3p_ft", "squad_ft", "ab3p_ft_proposal", "squad_ft_proposal",
           "ab3p_ft_rerank", "squad_ft_rerank", "ab3p_ft_proposal_rerank", "squad_ft_proposal_rerank"]
fns = [label_fn, ab3p_fn, bert_ab3p_ft_fn, bert_squad_ft_fn,
       bert_ab3p_ft_proposal_fn, bert_squad_ft_proposal_fn, bert_ab3p_ft_rerank_fn,
       bert_squad_ft_rerank_fn, bert_ab3p_ft_proposal_rerank_fn, bert_squad_ft_proposal_rerank_fn]
assert len(methods) == len(fns)


dfs = dict()
for method, fn in zip(methods, fns):
    if "rerank" in method:
        df = rerank2table(fn)
        df = df[df["comment"] != "ab3p"]
        df["score"] = df["score"].apply(lambda x: logit2prob(x))
        df["comment"] = 1
    elif "ft" in method:
        df = fasta2table(fn)
        df = df[df["comment"] != "ab3p"]
        df["comment"] = df["comment"].apply(
            lambda x: int(x.replace("bqaf", "")))
    elif method == "ab3p":
        df = fasta2table(fn)
        df["comment"] = 1
    elif method == "label":
        df = fasta2table(fn)

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
for method in ["ab3p", "ab3p_ft_proposal", "squad_ft_proposal",
               "ab3p_ft_proposal_rerank", "squad_ft_proposal_rerank"]:
    if ("rerank" in method) or (method == "ab3p"):
        df = dfs[method]
        intxn = pd.merge(label, df, how="outer", on=[
            "pmid", "type", "sent_no", "sent", "sf", "lf"], suffixes=("_label", "_pred"))
        intxn.score_label.fillna(0, inplace=True)
        intxn.score_pred.fillna(-999.0, inplace=True)
        intxn = intxn[["sf", "lf", "pmid", "type", "sent_no",
                       "sent", "score_label", "score_pred", "comment_label"]]
        intxns[method] = intxn
        precisions, recalls, scores = pr(intxn.score_pred, intxn.score_label)
        metrics["method"] += [method] * len(precisions)
        metrics["top"] += scores
        metrics["recall"] += recalls
        metrics["precision"] += precisions

    else:
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
            # print(f"{method} recall @ {k}: {recall:.3f}")
            # print(f"{method} precision @ {k}: {precision:.3f}")
            metrics["method"].append(method)
            metrics["top"].append(k)
            metrics["recall"].append(recall)
            metrics["precision"].append(precision)

metrics = pd.DataFrame(metrics)

plt.close()
plt.ion()
sns.lineplot(data=metrics, x="recall", y="precision", hue="method")
plt.show()
plt.savefig(f"{out_dir}/pr_curve.pdf")


intxns["squad_ft_proposal_rerank"].loc[lambda x: x["score_label"] == 0][["sf", "lf", "score_pred", "sent"]].to_csv(f"{out_dir}/squad_ft_proposal_rerank_overgenerated", sep="\t", index=False)
intxns["squad_ft_proposal_rerank"].loc[lambda x: x["score_pred"] == -999][["sf", "lf", "score_pred", "sent"]].to_csv(f"{out_dir}/squad_ft_proposal_rerank_missing", sep="\t", index=False)
intxns["squad_ft_proposal_rerank"].loc[lambda x: (x["score_pred"] != -999) & (x["score_label"] != 0)][["sf", "lf", "score_pred", "sent"]].to_csv(f"{out_dir}/squad_ft_proposal_rerank_correct", sep="\t", index=False)
intxns["squad_ft_proposal_rerank"].to_csv(f"{out_dir}/squad_ft_proposal_rerank", sep="\t", index=False)