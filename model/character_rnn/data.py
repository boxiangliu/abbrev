import os
import string
from unidecode import unidecode
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader


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


# from torch.utils.data import DataLoader
# from pathlib import Path
# data_dir = Path("../processed_data/preprocess/bioc/propose_on_bioc/")
# from model.character_rnn.data import SFData
# data1 = SFData([data_dir / "medstract"], one_hot=False)
# loader = DataLoader(data1, batch_size=2, shuffle=True, collate_fn=data1._pad_seq)
# tensors, labels, seq_lens, seqs = next(iter(loader))
# tensors.dtype
# labels
# seqs
# seq_lens
# dataset = SFLFPairs("test")
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True,
#                         num_workers=1, collate_fn=pad_seq)
# i, (labels, padded_seqs, seq_lens) = next(enumerate(dataloader))
