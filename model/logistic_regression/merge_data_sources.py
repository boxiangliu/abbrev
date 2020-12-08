from pathlib import Path
import pandas as pd

ab3p_dir = Path("../processed_data/model/qa_to_ab3p/parse_ab3p_output/")
datasets = ["Ab3P", "bioadi", "medstract", "SH"]

ab3p = {}
for dataset in datasets:
    ab3p[dataset] = pd.read_table(ab3p_dir / dataset)
