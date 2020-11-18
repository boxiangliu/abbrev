import torch
import glob
from unidecode import unidecode
import string
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.nn as nn


class ToyData(Dataset):
    """Create a dataset for the toy data"""

    def __init__(self, fn):
        self.data = self.read_toy_data(fn)
        self.letters = string.ascii_lowercase
        self.n_letters = len(self.letters)

    def __len__(self):
        return len(self.data["seq"])

    def __getitem__(self, idx):
        seq = self.seq2tensor(self.data["seq"][idx])
        label = int(self.data["label"][idx])
        return seq, label

    def read_toy_data(self, fn):
        container = {"seq": [], "label": []}
        with open(fn) as f:
            for line in f:
                split_line = line.strip().split("\t")
                container["seq"].append(split_line[0])
                container["label"].append(split_line[1])
        return container

    def seq2tensor(self, seq):
        tensor = torch.zeros(len(seq), self.n_letters)
        for i, letter in enumerate(seq):
            tensor[i][self.letters.index(letter)] = 1
        return tensor

    def _pad_seq(self, samples):
        seq, label = zip(*samples)
        seq_lens = [len(s) for s in seq]
        sorted_list = sorted(zip(seq, label, seq_lens), key=lambda x: -x[2])
        seq, label, seq_lens = zip(*sorted_list)
        seq = pad_sequence(seq)
        return seq, torch.tensor(label), torch.tensor(seq_lens)

    def pack_seq(self, samples):
        seqs, labels, seq_lens = self._pad_seq(samples)
        seqs = pack_padded_sequence(seqs, seq_lens)
        return seqs, labels, seq_lens


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


class MyDataParallel(nn.DataParallel):

    def __getattr__(self, name):
        return getattr(self.module, name)