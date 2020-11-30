import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class EmbedRNN(nn.Module):
    """Recurrent neural network"""

    def __init__(self, input_size, hidden_size, output_size, embed_size=16):
        """Args:
                input_size (int): input dimension of a time step.
                hidden_size (int): dimesion of hidden layer.
                output_size (int): number of output categories.  
        """
        super(EmbedRNN, self).__init__()

        print(f"input_size:{input_size}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed_size = embed_size

        self.dropout = nn.Dropout(p=0.5)
        self.embed = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size)
        self.fc1 = nn.Linear(in_features=hidden_size,
                            out_features=output_size)
        self.fc2 = nn.Linear(in_features=2 * hidden_size,
                            out_features=output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.pack_padded_sequence = pack_padded_sequence

    def forward(self, sfs, lfs, sf_lens, lf_lens):
        """Args:
                seqs (PackedSequence): Packed padded sequence.
        """
        sf_embedding = self.embed(sfs)
        lf_embedding = self.embed(lfs)
        sfs = self.pack_padded_sequence(sf_embedding, sf_lens.cpu(), enforce_sorted=False)
        lfs = self.pack_padded_sequence(lf_embedding, lf_lens.cpu(), enforce_sorted=False)

        sf_output, (sf_hidden, sf_cell) = self.rnn(sfs)
        lf_output, (lf_hidden, lf_cell) = self.rnn(lfs)

        sf_hidden = self.dropout(sf_hidden)
        lf_hidden = self.dropout(lf_hidden)

        sf_output = self.fc1(sf_hidden)
        # hidden[0] -> remove zeroth dimension
        pair_output = torch.cat(sf_hidden[0], lf_hidden[0], dim=1)
        pair_output = self.fc2(pair_output)

        return self.softmax(sf_output), self.softmax(pair_output)
