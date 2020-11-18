import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
sys.path.append(
    "/mnt/scratch/boxiang/projects/abbrev/scripts/model/character_rnn/example/")
from data import ToyData, WrappedDataLoader
from torch.utils.data import DataLoader
from model import RNN
import random
import time
import math
import pickle

hidden_size = 128
n_epochs = 5
print_every = 5000
plot_every = 1000
# If you set this too high, it might explode. If too low, it might not learn
learning_rate = 0.005
batch_size = 4
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
    names_dl = DataLoader(names_ds, batch_size=n_batch,
                          shuffle=True, num_workers=1, collate_fn=pad_seq)
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
                guess = torch.argmax(
                    rnn(lineToTensor(line).unsqueeze(1))).item()
                label = random_example[1]
                correct = '✓' if guess == label else '✗ (%s)' % label
                print('%d (%s) %.4f %s / %s %s' %
                      (n_steps, timeSince(start), loss, line, guess, correct))

            # Add current loss avg to list of losses
            if n_steps % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0

    torch.save(rnn, 'char-rnn-classification.pt')
    with open("./all_losses.pkl", "wb") as fout:
        pickle.dump(all_losses, fout)


def get_model(input_size, hidden_size, output_size, device):
    model = RNN(input_size, hidden_size, output_size).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9)
    return model, optimizer


def get_data(batch_size):
    data = ToyData(
        "../processed_data/model/character_rnn/example/toy_data/toy_data.tsv")
    assert len(data) == 10000
    return data, DataLoader(data, batch_size=batch_size, collate_fn=data.pad_seq)


def train_batch(model, loss_func, seqs, labels, seq_lens, opt=None):
    output = model(seqs, seq_lens)
    loss = loss_func(output, labels)
    batch_size = seqs.size()[1]

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), batch_size


def fit(n_epochs, model, loss_func, opt, train_loader, device, save_every=1000):
    n_steps = 0
    all_losses = []
    current_loss = 0
    start = time.time()

    for epoch in range(n_epochs):
        model.train()
        for seqs, labels, seq_lens in train_loader:
            n_steps += 1
            loss, n_examples = train_batch(
                model, loss_func, seqs, labels, seq_lens, opt)
            current_loss += loss

            if n_steps % save_every == 0:
                print('%d (%s) %.4f' %
                      (n_steps, timeSince(start), loss))
                all_losses.append(current_loss / save_every)
                current_loss = 0

    return all_losses

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
toy_data, toy_loader = get_data(batch_size)
input_size = output_size = toy_data.n_letters
model, opt = get_model(input_size, hidden_size, output_size, device)
loss_func = nn.NLLLoss()
fit(n_epochs, model, loss_func, opt, toy_loader, device)

n = 0
for seqs, labels, seq_lens in toy_loader:
    print(torch.argmax(model(seqs, seq_lens), dim=1))
    print(labels)
    n += 1
    if n > 3:
        break


if __name__ == "__main__":
    main()
