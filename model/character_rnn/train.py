import torch
import torch.nn as nn
import sys
sys.path.append("./model/character_rnn/")
from data import SFData, WrappedDataLoader
from torch.utils.data import DataLoader
from model import RNN
import time
import math
import pickle
from pathlib import Path

hidden_size = 128
n_epochs = 5
save_every = 1000
# If you set this too high, it might explode. If too low, it might not learn
learning_rate = 0.005
batch_size = 16


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def to_device(*args):
    return [x.to(device) for x in args]


def get_model(input_size, hidden_size, output_size, device):
    model = RNN(input_size, hidden_size, output_size).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9)
    return model, optimizer


def get_data(batch_size):
    data_dir = Path("../processed_data/preprocess/bioc/propose_on_bioc/")
    sf_eval = SFData([data_dir / "SH"])
    sf_train = SFData([data_dir / "Ab3P", data_dir / "bioadi",
                       data_dir / "medstract"], exclude=set(sf_eval.data["seq"]))
    return sf_train, DataLoader(sf_train, batch_size=batch_size,
                                collate_fn=sf_train.pack_seq), \
        sf_eval, DataLoader(sf_eval, batch_size=batch_size,
                            collate_fn=sf_eval.pack_seq)


def train_batch(model, loss_func, seqs, labels, seq_lens, opt=None):
    output = model(seqs, seq_lens)

    loss = loss_func(output, labels)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), batch_size


def fit(n_epochs, model, loss_func, opt, train_loader, save_every=1000):
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
train_data, train_loader, eval_data, eval_loader = get_data(batch_size)
toy_loader = WrappedDataLoader(toy_loader, to_device)
input_size = output_size = toy_data.n_letters
model, opt = get_model(input_size, hidden_size, output_size, device)
loss_func = nn.NLLLoss()
all_losses = fit(n_epochs, model, loss_func, opt, toy_loader)

torch.save(model, 'char-rnn-classification.pt')
with open("./all_losses.pkl", "wb") as fout:
    pickle.dump(all_losses, fout)
