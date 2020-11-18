import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence


seq1 = torch.ones(5, 3)
seq2 = torch.ones(4, 3)
seq_lens = [torch.tensor(x.size(0)) for x in [seq1, seq2]]

padded_seqs = pad_sequence([seq1, seq2])
packed_seqs = pack_padded_sequence(padded_seqs, seq_lens)
packed_seqs1 = pack_sequence([seq1, seq2])
assert packed_seqs == packed_seqs1


rnn = nn.RNN(input_size=3, hidden_size=1)
output, hidden = rnn(padded_seqs)
output.size()
hidden.size()

output1, hidden1 = rnn(packed_seqs)
output
output1
hidden
hidden1


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

    def pad_seq(self, samples):
        seq, label = zip(*samples)
        seq_lens = [len(s) for s in seq]
        sorted_list = sorted(zip(seq, label, seq_lens), key=lambda x: -x[2])
        seq, label, seq_lens = zip(*sorted_list)
        seq = pad_sequence(seq)
        return seq, torch.tensor(label), torch.tensor(seq_lens)

    def pack_seq(self, samples):
        seqs, labels, seq_lens = self.pad_seq(samples)
        seqs = pack_padded_sequence(seqs, seq_lens)
        return seqs, labels, seq_lens


import string
toy_data = ToyData(
    "../processed_data/model/character_rnn/example/toy_data/toy_data.tsv")
assert len(toy_data) == 10000
toy_loader = DataLoader(toy_data, batch_size=4, collate_fn=toy_data.pack_seq)
seqs, labels, seq_lens = next(iter(toy_loader))
