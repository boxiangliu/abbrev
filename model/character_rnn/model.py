import torch
import torch.nn as nn


class RNN(nn.Module):
    """Recurrent neural network"""

    def __init__(self, input_size, hidden_size, output_size, arch="rnn"):
        """Args:
                input_size (int): input dimension of a time step.
                hidden_size (int): dimesion of hidden layer.
                output_size (int): number of output categories.  
        """
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.arch = arch

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.5)

        if arch == "rnn":
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size)
        elif arch == "lstm":
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size)

        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, seqs):
        """Args:
                seqs (PackedSequence): Packed padded sequence.
        """
        seqs = nn.dropout1(seqs)
        if self.arch == "rnn":
            output, hidden = self.rnn(seqs)
        elif self.arch == "lstm":
            output, (hidden, cell) = self.rnn(seqs)

        hidden = self.dropout2(hidden)
        output = self.fc(hidden[0])

        return self.softmax(output)
