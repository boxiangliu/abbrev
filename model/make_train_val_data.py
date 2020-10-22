import pandas as pd
import os

nrows = 1e5
ab3p_fn = "../processed_data/preprocess/ab3p/summarize_ab3p/ab3p_res.csv"
out_dir = "../processed_data/preprocess/model/train_val/"
os.makedirs(out_dir, exist_ok=True)
ab3p = pd.read_csv(ab3p_fn, sep="\t", nrows=nrows)

train_pct=0.8
train_div = round(ab3p.shape[0] * train_pct)
train_idx = range(train_div)
val_idx = range(train_div, len(contexts))


ab3p.iloc[train_idx,:].to_csv(f"{out_dir}/train.tsv", sep="\t", index=False)
ab3p.iloc[val_idx,:].to_csv(f"{out_dir}/val.tsv", sep="\t", index=False)