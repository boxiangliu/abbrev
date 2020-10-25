import pandas as pd
import os

nrows = 1e6
ab3p_fn = "../processed_data/preprocess/ab3p/summarize_ab3p/ab3p_res.csv"
out_dir = "../processed_data/preprocess/model/train_val/"
os.makedirs(out_dir, exist_ok=True)
ab3p = pd.read_csv(ab3p_fn, sep="\t", nrows=nrows)

train_pct = 0.6
val_pct = 0.2
test_pct = 0.2
denom = sum([train_pct, val_pct, test_pct])
train_pct, val_pct, test_pct = \
    train_pct/denom, val_pct/denom, test_pct/denom

train_div = round(ab3p.shape[0] * train_pct)
val_div = train_div + round(ab3p.shape[0] * val_pct)
train_idx = range(train_div)
val_idx = range(train_div, val_div)
test_idx = range(val_div, ab3p.shape[0])
assert len(train_idx) + len(val_idx) + len(test_idx) == ab3p.shape[0]


ab3p.iloc[train_idx,:].to_csv(f"{out_dir}/train.tsv", sep="\t", index=False)
ab3p.iloc[val_idx,:].to_csv(f"{out_dir}/val.tsv", sep="\t", index=False)
ab3p.iloc[test_idx,:].to_csv(f"{out_dir}/test.tsv", sep="\t", index=False)