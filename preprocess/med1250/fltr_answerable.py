# from tqdm import tqdm
# import sys
# sys.path.append(".")
# from utils import create_dir_by_fn


# med1250_fn = "../processed_data/preprocess/med1250/text2fasta/MED1250_labeled"
# out_fn = "../processed_data/preprocess/med1250/fltr_answerable/MED1250_labeled"


# create_dir_by_fn(out_fn)

# with open(med1250_fn) as fin, open(out_fn, "w") as fout:

for line in sys.stdin:
    if line.startswith(">"):
        header = line
        new_line = True
    elif line.startswith("  "):
        answer = line
        if new_line:
            sys.stdout.write(header)
            sys.stdout.write(sentence)
            new_line = False
        sys.stdout.write(answer)
    else:
        sentence = line