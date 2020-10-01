#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --job-name=test1
#SBATCH --output=test1.out
python3 preprocess/split_sent.py --in_fn ../processed_data/preprocess/abstract/pubmed19n0971.txt --out_fn test971.txt