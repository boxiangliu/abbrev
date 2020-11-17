import torch
import glob
from unidecode import unidecode
import string
from torch.utils.data import Dataset, DataLoader


all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)


# Read a file and split into lines
def readLines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unidecode(line) for line in lines]


def read_file_pattern(fpattern='../../practical-pytorch/data/names/*.txt'):
    container = {"line": [], "category": []}
    n_categories = 0
    all_categories = []
    for fn in glob.glob(fpattern):
        category = fn.split('/')[-1].split('.')[0]
        n_categories += 1
        all_categories.append(category)
        lines = readLines(fn)
        for line in lines:
            container["line"].append(line)
            container["category"].append(category)
    return container, n_categories, all_categories


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors


def lineToTensor(line):
    tensor = torch.zeros(len(line), n_letters)
    for li, letter in enumerate(line):
        tensor[li][letterToIndex(letter)] = 1
    return tensor


def pad_seq(samples):
    """Pad sequences in a batch to the same maximum length"""
    seqs, labels = zip(*samples)

    seq_lens = [len(pair) for pair in seqs]
    max_len = max(seq_lens)
    sorted_list = sorted(zip(seqs, labels, seq_lens), key=lambda x: -x[2])
    seqs, labels, seq_lens = zip(*sorted_list)

    padded_seqs = torch.zeros(max_len, len(labels), n_letters)

    for (i, pair) in enumerate(seqs):
        padded_seqs[:len(pair), i, :] = lineToTensor(pair)

    return torch.tensor(labels), padded_seqs, seq_lens, seqs


class NamesData(Dataset):
    """Name and origin"""

    def __init__(self, fpattern):
        data, n_categories, all_categories = read_file_pattern(fpattern)
        self.data = data
        self.n_categories = n_categories
        self.all_categories = all_categories

    def __len__(self):
        return len(self.data["line"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        line = self.data["line"][idx]
        category = self.data["category"][idx]
        label = self.all_categories.index(category)

        return line, label


dataset = NamesData("../../practical-pytorch/data/names/*.txt")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn = pad_seq)
i, (labels, padded_seqs, seq_lens, seqs) = next(enumerate(dataloader))
