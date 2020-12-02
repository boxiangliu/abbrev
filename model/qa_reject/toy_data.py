import random
import string
import click
import sys

# N = 10000
n_words_per_lf, n_chars_per_sf = 2, 2
min_chars, max_chars = 2, 10
chars = string.ascii_uppercase + string.digits
n_chars = len(chars)


@click.command()
@click.option("--n_examples", type=int, help="Number of examples to generate.")
def main(n_examples):
    for n in range(n_examples):
        sys.stdout.write("sf\tlf\tlabel\n")
        SF, LF = gen_valid_pair(n_words_per_lf, min_chars,
                                max_chars, chars, n_chars)
        sys.stdout.write(f"{SF}\t{LF}\t1\n")
        rand_SF, rand_LF = gen_random_pair(
            n_words_per_lf, min_chars, max_chars, n_chars_per_sf, chars, n_chars)
        sys.stdout.write(f"{rand_SF}\t{rand_LF}\t0\n")


def gen_lf(n_words_per_lf, min_chars, max_chars, chars, n_chars):
    LF = []
    for w in range(n_words_per_lf):
        length = random.randint(min_chars, max_chars)
        word = "".join([chars[random.randint(0, n_chars - 1)]
                        for _ in range(length)])
        LF.append(word)
    return "\t".join(LF)


def lf_to_sf(lf):
    return [x[0] for x in lf.split("\t")]


def gen_sf(n_chars_per_sf, chars, n_chars):
    return "".join([chars[random.randint(0, n_chars - 1)] for _ in range(n_chars_per_sf)])


def gen_valid_pair(n_words_per_lf, min_chars, max_chars, chars, n_chars):
    LF = gen_lf(n_words_per_lf, min_chars, max_chars, chars, n_chars)
    SF = lf_to_sf(LF)
    return SF, LF


def gen_random_pair(n_words_per_lf, min_chars, max_chars, n_chars_per_sf, chars, n_chars):
    LF = gen_lf(n_words_per_lf, min_chars, max_chars, chars, n_chars)
    SF = lf_to_sf(LF)
    random_SF = SF
    while random_SF == SF:
        random_SF = gen_sf(n_chars_per_sf, chars, n_chars)
    return random_SF, LF

if __name__ == '__main__':
    main()
