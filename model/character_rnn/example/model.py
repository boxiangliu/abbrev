import torch
import torch.nn as nn
from torch.autograd import Variable


# class RNN(nn.Module):

#     def __init__(self, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()

#         self.hidden_size = hidden_size

#         self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
#         self.i2o = nn.Linear(input_size + hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=0)

#     def forward(self, input, hidden):
#         combined = torch.cat((input, hidden), 1)
#         hidden = self.i2h(combined)
#         output = self.i2o(combined)
#         output = self.softmax(output)
#         return output, hidden

#     def initHidden(self):
#         return Variable(torch.zeros(1, self.hidden_size))


# class RNN(nn.Module):
#     """Recurrent neural network"""

#     def __init__(self, n_letters, n_hidden, n_categories):
#         super(RNN, self).__init__()

#         self.n_hidden = n_hidden
#         self.n_letters = n_letters
#         self.n_categories = n_categories

#         self.rnn = nn.RNN(input_size=n_letters, hidden_size=n_hidden)
#         self.fc = nn.Linear(in_features=n_hidden, out_features=n_categories)
#         self.softmax = nn.LogSoftmax(dim=0)

#     def forward(self, input):
#         output, hidden = self.rnn(input)
#         n_batch = hidden.size()[1]
#         prob = self.fc(hidden.transpose(0, 1).reshape(n_batch, -1))
#         prob = self.softmax(prob)

#         return prob


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
                seqs (tensor): Tensor of dimensions T x B x S, 
                    where T is the length of the longest sequence, 
                    B is the batch size, S is the input_size.
                seq_lens (list or tuple): pre-padding sequence lengths used 
                    to select the appropriate timestep as output.
        """
        output, hidden = self.rnn(seqs)
        batch_size = output.size()[1]

        output = torch.cat([output[t - 1, b, :].unsqueeze(0)
                            for (t, b) in zip(seq_lens, range(batch_size))], dim=0)
        assert output.size() == torch.Size(batch_size, self.hidden_size)

        output = self.fc(output)
        return self.softmax(output)


