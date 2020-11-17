import torch
import sys
sys.path.append("./model/character_rnn/")
from data import *
from model import *
import random
import time
import math
import pickle
import torch.nn as nn


n_hidden = 128
n_epochs = 100000
print_every = 5000
plot_every = 1000
# If you set this too high, it might explode. If too low, it might not learn
learning_rate = 0.005


def train(category_tensor, line_tensor, rnn, optimizer, criterion):
    hidden = rnn.initHidden()
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
    if torch.cuda.device_count() > 1:
        sys.stderr.write(f"Running on {torch.cuda.device_count()} GPUs.")

    rnn = RNN(n_letters, n_hidden, n_categories)

    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()

    for epoch in range(1, n_epochs + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor, rnn, optimizer, criterion)
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
