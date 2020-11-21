import torch
import string
from unidecode import unidecode
import os
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from pathlib import Path
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


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


class SFData(Dataset):
    """Classify short form into valid and invalid"""

    def __init__(self, flist, exclude=set(), one_hot=True):
        """Args:
            flist (list): a list of files.
            exclude (set): a set of short forms to exclude.
                This can be used to remove short forms in eval set.
        """
        self.flist = flist
        self.data = self.read_files(flist, exclude)
        # empty string means unknown
        self.characters = [""] + list(string.printable)
        self.n_characters = len(self.characters)
        self.one_hot = one_hot

    def __len__(self):
        return len(self.data["seq"])

    def __getitem__(self, idx):
        seq = self.data["seq"][idx]
        if self.one_hot:
            tensor = self.seq2one_hot(seq)
        else:
            tensor = self.seq2idx(seq)

        label = int(self.data["label"][idx])
        return tensor, label, seq

    def read_files(self, flist, exclude):
        seqs = []
        labels = []
        for fn in flist:
            with open(fn) as f:
                for line in f:
                    split_line = line.strip().split("\t")
                    if len(split_line) == 2:
                        if split_line[0] not in exclude:
                            seqs.append(split_line[0])
                            labels.append(split_line[1])
        return {"seq": seqs, "label": labels}

    def seq2one_hot(self, seq):
        tensor = torch.zeros(len(seq), self.n_characters)
        for i, character in enumerate(seq):
            ascii_char = unidecode(character)
            tensor[i][self.characters.index(ascii_char)] = 1
        return tensor

    def seq2idx(self, seq):
        tensor = torch.zeros(len(seq), dtype=torch.long)
        for i, character in enumerate(seq):
            ascii_char = unidecode(character)
            tensor[i] = self.characters.index(ascii_char)
        return tensor

    def _pad_seq(self, samples):
        tensor, label, seq = zip(*samples)
        seq_lens = [len(s) for s in tensor]
        sorted_list = sorted(
            zip(tensor, label, seq_lens, seq), key=lambda x: -x[2])
        tensor, label, seq_lens, seq = zip(*sorted_list)
        tensor = pad_sequence(tensor)
        return tensor, torch.tensor(label), torch.tensor(seq_lens), seq

    def pack_seq(self, samples):
        tensors, labels, seq_lens, seqs = self._pad_seq(samples)
        tensors = pack_padded_sequence(tensors, seq_lens)
        return tensors, labels, seq_lens, seqs

    def one_hot2seq(self, tensor):
        return "".join([self.characters[j] for i, j in tensor.nonzero()])


class WrappedDataLoader:

    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


# dataset = SFLFPairs("test")
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True,
#                         num_workers=1, collate_fn=pad_seq)
# i, (labels, padded_seqs, seq_lens) = next(enumerate(dataloader))
