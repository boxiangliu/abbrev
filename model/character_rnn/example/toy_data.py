import sys
import random
import string

all_letters = string.ascii_lowercase
n_letters = len(all_letters)
min_len = 1
max_len = 10
n_examples = 10000
random.seed(42)
for i in range(n_examples):
    idx = random.randint(0, n_letters)
    chosen_letter = all_letters[idx]
    seq_len = random.randint(min_len, max_len)
    seq = chosen_letter * seq_len
    sys.stdout.write(f"{seq}\t{idx}\n")


