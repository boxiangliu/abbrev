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
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

hidden_size = 512
n_epochs = 15
save_every = 50
# If you set this too high, it might explode. If too low, it might not learn
learning_rate = 0.005
batch_size = 16
output_size = 2
arch = "lstm"
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


OUT_DIR = Path("../processed_data/model/character_rnn/lstm/run_01/")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def to_device(*args):
    return [x.to(device) for x in args]


def get_model(input_size, hidden_size, output_size, arch, device):
    model = RNN(input_size, hidden_size, output_size, arch).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9)
    return model, optimizer


def get_data(batch_size):
    data_dir = Path("../processed_data/preprocess/bioc/propose_on_bioc/")
    sf_eval = SFData([data_dir / "medstract"])
    sf_train = SFData([data_dir / "Ab3P", data_dir / "bioadi",
                       data_dir / "SH"], exclude=set(sf_eval.data["seq"]))
    return sf_train, DataLoader(sf_train, batch_size=batch_size, shuffle=True,
                                collate_fn=sf_train.pack_seq), \
        sf_eval, DataLoader(sf_eval, batch_size=batch_size * 4,
                            collate_fn=sf_eval.pack_seq)


def loss_batch(model, loss_func, seqs, labels, opt=None):
    output = model(seqs)
    pred = torch.argmax(output, dim=1)

    loss = loss_func(output, labels)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), pred


def fit(n_epochs, model, loss_func, opt, train_loader, eval_loader, save_every=1000):
    n_steps, train_loss, train_corrects = 0, 0, 0
    train_losses, train_accuracies, eval_losses, eval_accuracies = [], [], [], []
    start = time.time()

    for epoch in range(n_epochs):
        for seqs, labels, seq_lens in train_loader:

            model.train()
            n_steps += 1
            loss, pred = loss_batch(
                model, loss_func, seqs, labels, opt)
            train_loss += loss
            train_corrects += sum(pred == labels)

            if n_steps % save_every == 0:

                avg_train_loss = train_loss / save_every
                train_losses.append(avg_train_loss)
                train_accuracy = train_corrects / save_every
                train_accuracies.append(train_accuracy)
                train_loss, train_corrects = 0, 0

                model.eval()
                eval_loss, eval_corrects = 0, 0
                with torch.no_grad():
                    for seqs, labels, seq_lens in eval_loader:
                        loss, pred = loss_batch(model, loss_func, seqs, labels)
                        eval_loss += loss
                        eval_corrects += sum(pred == labels)
                avg_eval_loss = eval_loss / len(eval_loader)
                eval_accuracy = eval_corrects / len(eval_loader)
                eval_losses.append(avg_eval_loss)
                eval_accuracies.append(eval_accuracy)

                print('STEP %d (%s) TRAIN=%.4f ACC=%.4f; EVAL=%.4f ACC=%.4f' %
                      (n_steps, timeSince(start), avg_train_loss, train_accuracy, avg_eval_loss, eval_accuracy))

    return train_losses, eval_losses, train_accuracies, eval_accuracies


def plot_metrics(train_losses, eval_losses, train_accuracies, eval_accuracies):
    eval_losses_smooth = savgol_filter(eval_losses, 15, 3)
    eval_accuracies_smooth = savgol_filter(eval_accuracies, 15, 3)

    plt.close()
    plt.plot(train_losses, label="train")
    plt.plot(eval_losses, label="eval")
    plt.plot(eval_losses_smooth, label="smooth eval")
    plt.title("Cross Entropy Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig(OUT_DIR / "losses.png")

    plt.close()
    plt.plot(train_accuracies, label="train")
    plt.plot(eval_accuracies, label="eval")
    plt.plot(eval_accuracies_smooth, label="smooth eval")
    plt.title("Accuracy")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.savefig(OUT_DIR / "accuracies.png")


def save_metrics(metrics):
    with open(OUT_DIR / "all_losses.pkl", "wb") as fout:
        pickle.dump(metrics, fout)


train_data, train_loader, eval_data, eval_loader = get_data(batch_size)
train_loader = WrappedDataLoader(train_loader, to_device)
eval_loader = WrappedDataLoader(eval_loader, to_device)
input_size = train_data.n_characters
model, opt = get_model(input_size, hidden_size, output_size, arch, device)
loss_func = nn.NLLLoss()
train_losses, eval_losses, train_accuracies, eval_accuracies = fit(
    n_epochs, model, loss_func, opt, train_loader, eval_loader, save_every)


plot_metrics(train_losses, eval_losses, train_accuracies, eval_accuracies)
torch.save(model, OUT_DIR / 'model.pt')
save_metrics([train_losses, eval_losses, train_accuracies, eval_accuracies])


for seq, label in eval_data:
    model(seq.unsqueeze(1).to(device))
seq.unsqueeze(1).to(device).size()
