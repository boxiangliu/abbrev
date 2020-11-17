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


class RNN(nn.Module):
    """Recurrent neural network"""

    def __init__(self, n_letters, n_hidden, n_categories, n_layers=1, bidirectional=False):
        super(RNN, self).__init__()

        self.n_directions = 1 if bidirectional == False else 2
        self.rnn = nn.RNN(input_size=n_letters, hidden_size=n_hidden,
                          num_layers=n_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(input_size=n_hidden, output_size=n_categories)
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, input):
        output, hidden = self.rnn(input)
        n_batch = hidden.size()[1]
        prob = self.fc(hidden.transpose(0,1).reshape(n_bath, self.n_directions * self.n_layers))
        prob = self.softmax(prob)

        return prob