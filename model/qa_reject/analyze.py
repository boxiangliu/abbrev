data_fn = "../processed_data/model/qa_reject/lstm/run_01/preds.tsv"
import pandas as pd
data = pd.read_table(data_fn)
data.loc[lambda x: x["sf_label"] == 1].groupby("sf").nlargest(1, "pr_prob")
# 136 rows in total

data.loc[lambda x: x["sf_label"] == 1].groupby("sf").apply(lambda x: x.nlargest(1, "pr_prob")).pr_label.sum()
# 79 correct

