import os
from collections import Counter
import glob

ab3p_dir = "../processed_data/preprocess/model/predict/ab3p_ft_data_1M/"
squad_dir = "../processed_data/preprocess/model/predict/squad_ft_data_1M/"
out_dir = "../processed_data/analysis/ab3p5bert/"
os.makedirs(out_dir, exist_ok=True)

def get_candidate_rank_freq(res_dir):
    candidate_ranks = []
    bert_match = None
    for fn in glob.glob(f"{res_dir}/*.out"):
        print(f"Input\t{fn}")
        with open(fn) as f:
            for line in f:
                if line.startswith("  "):
                    split_line = line.strip().split("|")
                    if split_line[0] == "ab3p":
                        ab3p = split_line[2]
                        if bert_match == False:
                            candidate_ranks.append("NOT_FOUND")
                        bert_match = False

                    elif bert_match == False:
                        if split_line[0].startswith("bqaf"):
                            if split_line[2] == ab3p:
                                candidate_rank = split_line[0].replace("bqaf", "")
                                candidate_ranks.append(candidate_rank)
                                bert_match = True

    candidate_rank_freq = Counter(candidate_ranks)
    return candidate_rank_freq

ab3p_candidate_rank_freq = get_candidate_rank_freq(ab3p_dir)
squad_candidate_rank_freq = get_candidate_rank_freq(squad_dir)


def save_candidate_rank_freq(candidate_rank_freq, out_fn):
    with open(out_fn, "w") as f:
        for k in ["1", "2", "3", "4" ,"5", "NOT_FOUND"]:
            f.write(f"CANDIDATE {k}\t{candidate_rank_freq[k]}\n")


save_candidate_rank_freq(ab3p_candidate_rank_freq, out_fn=f"{out_dir}/ab3p.txt")
save_candidate_rank_freq(squad_candidate_rank_freq, out_fn=f"{out_dir}/squad.txt")