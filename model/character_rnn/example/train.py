import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
sys.path.append("/mnt/scratch/boxiang/projects/abbrev/scripts/model/character_rnn/example/")
from data import *
import random
import time
import math
import pickle

n_hidden = 128
n_epochs = 100000
print_every = 5000
plot_every = 1000
# If you set this too high, it might explode. If too low, it might not learn
learning_rate = 0.005
n_batch = 4
n_layers = 1
bidirectional = False
n_directions = 1 if bidirectional == False else 2

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1)  # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor(
        [all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor



def train(category_tensor, line_tensor):
    optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    # return output, loss.data[0]
    return output, loss.data  # Boxiang


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def main():
    names_ds = NamesData(fpattern='../../practical-pytorch/data/names/*.txt')
    names_dl = DataLoader(names_ds, batch_size=n_batch, shuffle=True, num_workers=1, collate_fn=pad_seq)
    rnn = nn.RNN(input_size=n_letters, hidden_size=n_hidden, num_layers=n_layers, bidirectional=bidirectional, batch_first=True)
    ff = nn.Linear(n_hidden, names_ds.n_categories)
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()


    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    hidden = torch.zeros(n_batch, n_layers * n_directions, n_hidden, requires_grad=True)
    start = time.time()

    for epoch in range(1, n_epochs + 1):
        for labels, padded_seqs, _, _ in names_dl:
            output, hidden = rnn(padded_seqs, hidden)

        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

        # Print epoch number, loss, name and guess
        if epoch % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch /
                                                    n_epochs * 100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    torch.save(rnn, 'char-rnn-classification.pt')
    with open("./all_losses.pkl", "wb") as fout:
        pickle.dump(all_losses, fout)


if __name__ == "__main__":
    main()
