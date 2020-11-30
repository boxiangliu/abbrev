import os
import string
from unidecode import unidecode
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader


class SFLFData(Dataset):
    """Short form and long form dataset.
        Each example include the following items:
        1. short form
        2. long form
        3. whether the short form is valid
        4. whether the long form is valid
        5. whether the example is from the gold standard
    """

    def __init__(self, flist, exclude=set(), one_hot=True):
        """Args:
            flist (list): a list of files.
            exclude (set): a set of short forms to exclude.
                This can be used to remove short forms in eval set.
        """
        self.flist = flist
        self.data = self.read_files(flist, exclude)
        self.sf_lf_pairs = set([(sf, lf)
                                for sf, lf in zip(self.data["sf"], self.data["lf"])])
        # empty string means unknown
        self.characters = [""] + list(string.printable)
        self.n_characters = len(self.characters)
        self.one_hot = one_hot

    def __len__(self):
        return len(self.data["sf"])

    def __getitem__(self, idx):
        sf = self.data["sf"][idx]
        lf = self.data["lf"][idx]
        if self.one_hot:
            sf_tensor, lf_tensor = self.seq2one_hot(sf), self.seq2one_hot(lf)
        else:
            sf_tensor, lf_tensor = self.seq2idx(sf), self.seq2idx(lf)

        sf_label = self.data["sf_label"][idx]
        pair_label = self.data["pair_label"][idx]
        is_gold = self.data["is_gold"][idx]

        return sf_tensor, lf_tensor, sf_label, \
            pair_label, is_gold, sf, lf

    def read_files(self, flist, exclude):
        sfs, lfs, sf_labels, pair_labels, is_golds = [], [], [], [], []
        for fn in flist:
            with open(fn) as f:
                for line in f:
                    if line.startswith("sf\tlf\tgood_sf"):
                        continue

                    sf, lf, sf_label, pair_label, is_gold = \
                        line.strip().split("\t")

                    if (sf, lf) not in exclude:
                        sfs.append(sf)
                        lfs.append(lf)
                        sf_labels.append(int(sf_label))
                        pair_labels.append(int(pair_label))
                        is_golds.append(int(is_gold))

        assert len(sfs) == len(lfs)
        return {"sf": sfs, "lf": lfs, "sf_label": sf_labels,
                "pair_label": pair_labels, "is_gold": is_golds}

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
        sf_tensors, lf_tensors, sf_labels, pair_labels, \
            is_golds, sfs, lfs = zip(*samples)
        sf_lens = [len(s) for s in sf_tensors]
        lf_lens = [len(s) for s in lf_tensors]
        sf_tensors = pad_sequence(sf_tensors)
        lf_tensors = pad_sequence(lf_tensors)
        return sf_tensors, lf_tensors, torch.tensor(sf_labels), \
            torch.tensor(pair_labels), torch.tensor(is_golds), \
            torch.tensor(sf_lens), torch.tensor(lf_lens), \
            sfs, lfs

    def pack_seq(self, samples):
        sf_tensors, lf_tensors, sf_labels, pair_labels, is_golds, \
            sf_lens, lf_lens, sfs, lfs = self._pad_seq(samples)
        sf_tensors = pack_padded_sequence(
            sf_tensors, sf_lens, enforce_sorted=False)
        lf_tensors = pack_padded_sequence(
            lf_tensors, lf_lens, enforce_sorted=False)
        return sf_tensors, lf_tensors, sf_labels, \
            pair_labels, is_golds, \
            sf_lens, lf_lens, sfs, lfs

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
