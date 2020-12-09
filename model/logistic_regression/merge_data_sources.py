from pathlib import Path
import pandas as pd

ab3p_dir = Path("../processed_data/model/qa_to_ab3p/parse_ab3p_output/")
datasets = ["Ab3P", "bioadi", "medstract", "SH"]

ab3p = {}
for dataset in datasets:
    print(dataset)
    ab3p[dataset] = pd.read_table(ab3p_dir / dataset, quoting=3, dtype={"sent_no": "Int64"}, converters={"pmid": str}, na_values="")


def combine_dfs(dfs):
    df_list = []
    for source, df in dfs.items():
        df["source"] = source
        df_list.append(df)
    return pd.concat(df_list)


ab3p = combine_dfs(ab3p)

suffix_freq_fn = "../processed_data/model/suffix_freq/parse_suffix_freqs/suffix_freqs.tsv"
freq = pd.read_table(suffix_freq_fn, quoting=3)
test = pd.merge(ab3p, freq, on=["sf", "lf"], how="inner")
test.head(50)