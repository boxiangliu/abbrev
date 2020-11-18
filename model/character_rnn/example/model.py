import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    """Recurrent neural network"""

    def __init__(self, input_size, hidden_size, output_size):
        """Args:
                input_size (int): input dimension of a time step.
                hidden_size (int): dimesion of hidden layer.
                output_size (int): number of output categories.  
        """
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, seqs, seq_lens):
        """Args:
                seqs (PackedSequence): Packed padded sequence.
                seq_lens (list or tuple): pre-padding sequence lengths used 
                    to select the appropriate timestep as output.
        """
        output, hidden = self.rnn(seqs)
        output = self.fc(hidden.squeeze())
        return self.softmax(output)


