import torch
import glob
import string
from unidecode import unidecode
import os

all_letters = string.printable
n_letters = len(all_letters)


# Read a file and split into lines
def readLines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unidecode(line) for line in lines]


# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in glob.glob('../processed_data/preprocess/character_rnn/data/train/*'):
    category = 1 if "pos" in os.path.basename(filename) else 0
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor
