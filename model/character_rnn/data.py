import torch
import string
from unidecode import unidecode
import os
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


all_letters = string.printable
n_letters = len(all_letters)


def read_file(fn):
    container = defaultdict(list)
    with open(fn) as f:
        for line in f:
            split_line = line.strip().split("\t")
            sf = unidecode(split_line[0])
            lf = unidecode(split_line[1])
            container["sf"].append(sf)
            container["lf"].append(lf)
            container["pair"].append(f"{sf}\t{lf}")
            container["label"].append(int(split_line[2]))
    return container


def letterToIndex(letter):
    """Find letter index from all_letters, e.g. "a" = 0"""
    return all_letters.find(letter)


def indexToLetter(index):
    """Convert index to letters"""
    return all_letters[index]


def lineToTensor(line):
    """
    Turn a line into a <line_length x 1 x n_letters>,
        or an array of one-hot letter vectors
    """
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


class SFLFPairs(Dataset):
    """Short form and long form pairs"""

    def __init__(self, tsv_fn, transforms=None):
        """
        Args:
            tsv_fn (string): Path to tsv file with labels
            transform (callable, optional): Optional 
                transform to be applied on a sample.
        """
        self.tsv_fn = tsv_fn
        self.data = read_file(tsv_fn)

    def __len__(self):
        return len(self.data["sf"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pair = self.data["pair"][idx]
        label = self.data["label"][idx]

        return pair, label


def pad_seq(samples):
    """Pad sequences in a batch to the same maximum length"""
    pairs, labels = zip(*samples)

    seq_lens = [len(pair) for pair in pairs]
    max_len = max(seq_lens)
    sorted_list = sorted(zip(pairs, labels, seq_lens), key=lambda x: -x[2])
    pairs, labels, seq_lens = zip(*sorted_list)

    padded_seqs = torch.zeros(len(labels), max_len, 1, n_letters)

    for (i, pair) in enumerate(pairs):
        padded_seqs[i, :len(pair), :, :] = lineToTensor(pair)

    return torch.tensor(labels), padded_seqs, seq_lens


dataset = SFLFPairs("test")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn = pad_seq)
i, (labels, padded_seqs, seq_lens) = next(enumerate(dataloader))

