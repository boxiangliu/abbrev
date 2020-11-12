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
import numpy as np


label_fn = "../processed_data/preprocess/med1250/fltr_answerable/MED1250_labeled"
ab3p_fn = "../processed_data/evaluate/MED1250/MED1250_ab3p"
ab3p_ft_proposal_fn = "../processed_data/evaluate/MED1250/MED1250_bert-ab3p-ft_proposal"
squad_ft_proposal_fn = "../processed_data/evaluate/MED1250/MED1250_bert-squad-ft_proposal"

# Reranked:
ken_dir = "/mnt/big/kwc/pubmed/Boxiang/rerank/"
ab3p_ft_proposal_rerank_fn = f"{ken_dir}/MED1250_bert-ab3p-ft_proposal.rerank"
squad_ft_proposal_rerank_fn = f"{ken_dir}/MED1250_bert-squad-ft_proposal.rerank"

# Best
# A new method from Ken
squad_ft_proposal_best_fn = "/mnt/big/kwc/pubmed/Boxiang/forR/med1250/part3/MED1250_bert-squad-ft_all_proposal_nonredundant.best"
squad_ft_proposal_best_reject_fn = "../processed_data/evaluate/MED1250/MED1250_bert-squad-ft_all_proposal_nonredundant.best.filtered"
squad_ft_proposal_best_fit_fn = "/mnt/big/kwc/pubmed/Boxiang/forR/med1250/part3/MED1250_bert-squad-ft_all_proposal_nonredundant.best.0.fit2"


# Removed punctuation from the left edge:
rm_punc_dir = "../processed_data/model/rm_punc/"
ab3p_ft_proposal_rerank_rmPunc_fn = f"{rm_punc_dir}/MED1250_bert-ab3p-ft_proposal_rm-punc"
squad_ft_proposal_rerank_rmPunc_fn = f"{rm_punc_dir}/MED1250_bert-squad-ft_proposal_rm-punc"


# Reject scores:
reject_score_fn = "../processed_data/model/propose/MED1250_proposal_reject-score"

# Short form frequency:
freq_fn = "../processed_data/model/propose/MED1250_proposal_freq"


# Out dir:
out_dir = "../processed_data/evaluate/med1250/"
os.makedirs(out_dir, exist_ok=True)


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
    pairs = [(s, l) for s, l in sorted(
        zip(scores, labels), key=lambda x: x[0], reverse=True)]
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
            precision = n_correct_predictions / n_total_predictions
            precisions.append(precision)
            recall = n_correct_predictions / n_truths
            recalls.append(recall)
            scores.append(score)

    return precisions, recalls, scores


def plot_metrics(metrics):
    plt.close()
    plt.ion()
    sns.lineplot(data=metrics, x="recall", y="precision", hue="method")
    plt.show()


reject_score = pd.read_csv(reject_score_fn, sep="\t").drop_duplicates()
freq = pd.read_csv(freq_fn, sep="\t").drop_duplicates()


methods = ["label", "ab3p", "ab3p_ft_proposal", "squad_ft_proposal",
           "ab3p_ft_proposal_rerank", "squad_ft_proposal_rerank",
           "ab3p_ft_proposal_rerank_rmPunc", "squad_ft_proposal_rerank_rmPunc",
           "squad_ft_proposal_best", "squad_ft_proposal_best_reject", "squad_ft_proposal_best_fit"]
fns = [label_fn, ab3p_fn, ab3p_ft_proposal_fn, squad_ft_proposal_fn,
       ab3p_ft_proposal_rerank_fn, squad_ft_proposal_rerank_fn,
       ab3p_ft_proposal_rerank_rmPunc_fn, squad_ft_proposal_rerank_rmPunc_fn,
       squad_ft_proposal_best_fn, squad_ft_proposal_best_reject_fn, squad_ft_proposal_best_fit_fn]
assert len(methods) == len(fns)


dfs = dict()
for method, fn in zip(methods, fns):
    if "rerank" in method:
        df = rerank2table(fn)
        df = df[df["comment"] != "ab3p"]
        df["score"] = df["score"].apply(lambda x: logit2prob(x))
        df["comment"] = 1
    elif "best" in method:
        df = pd.read_csv(fn, sep="\t", quotechar=None, quoting=3).loc[lambda x: (x["newRank"] == 0) & (x["PMID"] != "PMID.1")]
        df = df.assign(pmid=df["PMID"].apply(lambda x: int(x.split("|")[0])), 
            type=df["PMID"].apply(lambda x: x.split("|")[1]),
            sent_no=df["PMID"].apply(lambda x: int(x.split("|")[2]))).drop(columns="PMID")
        if "fit" in method:
            df.rename(columns={"fit": "score", "SF": "sf", "LF": "lf", "newRank": "comment"}, inplace=True)
            df = df[["pmid", "type", "sent_no", "score", "sf", "lf", "comment"]]
        else: 
            df.rename(columns={"newScore": "score", "SF": "sf", "LF": "lf", "newRank": "comment"}, inplace=True)

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

    # if "proposal" in method:
    #     df_reject = pd.merge(df, reject_score[["sf", "reject_score"]], how="inner", on="sf")
    #     df_reject = df_reject.loc[lambda x: x["reject_score"] > -2]
    #     dfs[f"{method}_reject"] = df_reject

    #     df_freq = pd.merge(df, freq, how="inner", on="sf")
    #     df_freq = df_freq.loc[lambda x: x["freq"] > 0]
    #     dfs[f"{method}_freq"] = df_freq


########
# Help #
########
# Output: Recall
# Definition: ({sf-lf pair in ab3p} intersect {sf-lf pairs in label}) /
# {sf-lf pairs in label}
label = dfs["label"]
intxns = dict()
metrics = defaultdict(list)
for method in ["ab3p",  # "ab3p_ft_proposal", "squad_ft_proposal",
               # "ab3p_ft_proposal_reject", "squad_ft_proposal_reject",
               # "ab3p_ft_proposal_rerank",
               # "squad_ft_proposal_rerank",
               # "ab3p_ft_proposal_rerank_reject",
               # "squad_ft_proposal_rerank_reject",
               # "ab3p_ft_proposal_rerank_rmPunc",
               "squad_ft_proposal_rerank_rmPunc",
               # "ab3p_ft_proposal_rerank_rmPunc_reject",
               # "squad_ft_proposal_rerank_rmPunc_reject",
               # "squad_ft_proposal_rerank_rmPunc_freq",
               "squad_ft_proposal_best",
               "squad_ft_proposal_best_reject",
               "squad_ft_proposal_best_fit"]:
    if ("rerank" in method) or (method == "ab3p") or ("best" in method):
        df = dfs[method]
        if "best" in method:
            intxn = pd.merge(label, df, how="outer", on=[
                "pmid", "type", "sent_no", "sf", "lf"], suffixes=("_label", "_pred"))
        else:
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
plot_metrics(metrics)
plt.savefig(f"{out_dir}/pr_curve.pdf")



temp = pd.merge(label, df, how="left", on=[
    "pmid", "type", "sent_no", "sf", "lf"], suffixes=("_label", "_pred"))
temp.loc[temp["comment_pred"].isna()]


temp3 = pd.merge(label, dfs["squad_ft_proposal_rerank_rmPunc"], how="left", on=[
    "pmid", "type", "sent_no", "sf", "lf"], suffixes=("_label", "_pred"))
temp3.loc[temp3["comment_pred"].isna()]



intxns["squad_ft_proposal_rerank_rmPunc"].loc[lambda x: x["score_pred"] == -999][["sf", "lf", "score_pred", "sent"]].to_csv(f"{out_dir}/squad_ft_proposal_rerank_rmPunc_missing", sep="\t", index=False)
intxns["squad_ft_proposal_rerank_rmPunc"].loc[lambda x: (x["score_pred"] != -999) & (x["score_label"] != 0)][["sf", "lf", "score_pred", "sent"]].to_csv(f"{out_dir}/squad_ft_proposal_rerank_rmPunc_correct", sep="\t", index=False)
intxns["squad_ft_proposal_rerank_rmPunc"].to_csv(f"{out_dir}/squad_ft_proposal_rmPunc_rerank", sep="\t", index=False)
intxns["squad_ft_proposal_best_fit"].to_csv(f"{out_dir}/squad_ft_proposal_best_fit", sep="\t", index=False)

# Distinguish over-generation (SF does not exist in gold) and
# mismatch (SF exists in gold but LF does not match gold).
intxns["squad_ft_proposal_rerank_rmPunc"].groupby(["pmid", "type", "sent_no", "sent", "sf"]).agg(score_label=("score_label", "max")).reset_index().loc[lambda x: x["score_label"] == 0][["sf", "sent"]].to_csv(f"{out_dir}/squad_ft_proposal_rerank_rmPunc_overgenerated", sep="\t", index=False)
