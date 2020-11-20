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

        self.dropout = nn.Dropout(p=0.5)

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
        if self.arch == "rnn":
            output, hidden = self.rnn(seqs)
        elif self.arch == "lstm":
            output, (hidden, cell) = self.rnn(seqs)

        hidden = self.dropout(hidden)
        output = self.fc(hidden[0])

        return self.softmax(output)


class BERT(nn.Module):
    """Transformer encoder side"""
    def __init__(self, input_size, hidden_size, output_size):
        """Args:
                input_size (int): input dimension of a time step.
                hidden_size (int): dimesion of hidden layer.
                output_size (int): number of output categories.
        """
        super(BERT, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=8, dim_feedforward=hidden_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.decoder = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, seqs):
        output = self.encoder(seqs)
        output = self.decoder(output[0]) # the [CLS] token
        return self.softmax(output)



