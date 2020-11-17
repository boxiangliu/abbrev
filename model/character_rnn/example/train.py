import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
sys.path.append("/mnt/scratch/boxiang/projects/abbrev/scripts/model/character_rnn/example/")
from data import *
from model import RNN
import random
import time
import math
import pickle

n_hidden = 128
n_epochs = 50
print_every = 5000
plot_every = 1000
# If you set this too high, it might explode. If too low, it might not learn
learning_rate = 0.005
n_batch = 4
n_layers = 1
bidirectional = False


# def categoryFromOutput(output):
#     top_n, top_i = output.data.topk(1)  # Tensor out of Variable with .data
#     category_i = top_i[0][0]
#     return all_categories[category_i], category_i


# def randomChoice(l):
#     return l[random.randint(0, len(l) - 1)]


# def randomTrainingExample():
#     category = randomChoice(all_categories)
#     line = randomChoice(category_lines[category])
#     category_tensor = Variable(torch.LongTensor(
#         [all_categories.index(category)]))
#     line_tensor = Variable(lineToTensor(line))
#     return category, line, category_tensor, line_tensor



def train(labels, padded_seqs, rnn):
    optimizer.zero_grad()
    prob = rnn(padded_seqs)
    loss = criterion(prob, labels)
    loss.backward()
    optimizer.step()

    return prob, loss.data  # Boxiang


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def main():
    names_ds = NamesData(fpattern='../../practical-pytorch/data/names/*.txt')
    names_dl = DataLoader(names_ds, batch_size=n_batch, shuffle=True, num_workers=1, collate_fn=pad_seq)
    rnn = RNN(n_letters, n_hidden, names_ds.n_categories)
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    start = time.time()
    n_steps = 0
    for epoch in range(1, n_epochs + 1):
        for labels, padded_seqs, _, _ in names_dl:
            n_steps += 1
            output, loss = train(labels, padded_seqs, rnn)
            current_loss += loss

            # Print epoch number, loss, name and guess
            if n_steps % print_every == 0:
                random_example = names_ds[n_steps % len(names_ds)]
                line = random_example[0]
                guess = torch.argmax(rnn(lineToTensor(line).unsqueeze(1))).item()
                label = random_example[1]
                correct = '✓' if guess == label else '✗ (%s)' % label
                print('%d (%s) %.4f %s / %s %s' % (n_steps, timeSince(start), loss, line, guess, correct))

            # Add current loss avg to list of losses
            if n_steps % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0

    torch.save(rnn, 'char-rnn-classification.pt')
    with open("./all_losses.pkl", "wb") as fout:
        pickle.dump(all_losses, fout)


if __name__ == "__main__":
    main()
