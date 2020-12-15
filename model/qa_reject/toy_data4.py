import random
import string
import click
import sys

# N = 10000
min_words, max_words = 2, 10
min_chars, max_chars = 2, 10
chars = string.ascii_letters
n_chars = len(chars)
prob = 0.8

@click.command()
@click.option("--n_examples", type=int, help="Number of examples to generate.")
def main(n_examples):
    random.seed(42)
    sys.stdout.write("sf\tlf\tlabel\n")
    for n in range(n_examples):
        SF, LF = gen_valid_pair(min_words, max_words, min_chars,
                                max_chars, chars, n_chars)
        sys.stdout.write(f"{SF}\t{LF}\t1\n")
        rand_SF, rand_LF = gen_random_pair(
            min_words, max_words, min_chars, max_chars, chars, n_chars)
        sys.stdout.write(f"{rand_SF}\t{rand_LF}\t0\n")


def gen_lf(min_words, max_words, min_chars, max_chars, chars, n_chars):
    LF = []
    n_words_per_lf = random.randint(min_words, max_words)
    for w in range(n_words_per_lf):
        length = random.randint(min_chars, max_chars)
        word = "".join([chars[random.randint(0, n_chars - 1)]
                        for _ in range(length)])
        LF.append(word)
    return " ".join(LF)


def lf_to_sf(lf, prob):
    return "".join([random.choice(x).upper() for x in lf.split(" ") if random.random() < prob])


def gen_sf(n_chars_per_sf, chars, n_chars, prob):
    return "".join([chars[random.randint(0, n_chars - 1)].upper() for _ in range(n_chars_per_sf) if random.random() < prob])


def gen_valid_pair(min_words, max_words, min_chars, max_chars, chars, n_chars):
    LF = gen_lf(min_words, max_words, min_chars, max_chars, chars, n_chars)
    SF = lf_to_sf(LF, prob)
    return SF, LF


def gen_random_pair(min_words, max_words, min_chars, max_chars, chars, n_chars):
    LF = gen_lf(min_words, max_words, min_chars, max_chars, chars, n_chars)
    n_words = len(LF.split(" "))
    random_SF = gen_sf(n_words, chars, n_chars, prob)
    return random_SF, LF

if __name__ == '__main__':
    main()
