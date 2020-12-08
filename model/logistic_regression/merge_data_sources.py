from pathlib import Path
import pandas as pd

ab3p_dir = Path("../processed_data/model/qa_to_ab3p/parse_ab3p_output/")
datasets = ["Ab3P", "bioadi", "medstract", "SH"]

ab3p = {}
for dataset in datasets:
    print(dataset)
    ab3p[dataset] = pd.read_table(ab3p_dir / dataset)

def combine_dfs(dfs):
    df_list = []
    for source, df in dfs.items():
        df["source"] = source
        df_list.append(df)
    return pd.concat(df_list)


ab3p = combine_dfs(ab3p)