import pandas as pd
import os
import click

# nrows = 1e6
# ab3p_fn = "../processed_data/preprocess/ab3p/summarize_ab3p/ab3p_res.csv"
# out_dir = "../processed_data/preprocess/model/train_val/"
# train_pct = 0.6
# val_pct = 0.2
# test_pct = 0.2

@click.command()
@click.option("--nrows", type=int, help="Number of rows.")
@click.option("--ab3p_fn", type=str, help="Path to ab3p file.")
@click.option("--out_dir", type=str, help="Path to out directory.")
@click.option("--train_pct", type=float, help="Proportion used for training.", default=0.6)
@click.option("--val_pct", type=float, help="Proportion used for evaluation.", default=0.2)
@click.option("--test_pct", type=float, help="Proportion used for testing.", default=0.2)
def main(nrows, ab3p_fn, out_dir, train_pct, val_pct, test_pct):
    os.makedirs(out_dir, exist_ok=True)
    ab3p = pd.read_csv(ab3p_fn, sep="\t", nrows=nrows)

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


if __name__ == "__main__":
    main()