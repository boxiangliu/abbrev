import torch
import torch.nn as nn
import sys
sys.path.insert(0, "./model/character_rnn/")
from data import SFData, WrappedDataLoader
from torch.utils.data import DataLoader
from model import RNN, EmbedRNN
import time
import math
import pickle
from pathlib import Path
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import json
import click

# config_fn = "../processed_data/model/character_rnn/lstm/run_01/config.json"


@click.command()
@click.option("--config_fn", type=str, help="Path to configuration file.")
def main(config_fn):
    torch.manual_seed(42)
    config = read_config(config_fn)
    hidden_size, n_epochs, save_every, learning_rate, batch_size, \
        output_size, embed_size, train_sets, eval_sets, arch = set_config(config)

    train_data, train_loader, eval_data, eval_loader = get_data(
        batch_size, train_sets, eval_sets, arch)
    train_loader = WrappedDataLoader(train_loader, to_device)
    eval_loader = WrappedDataLoader(eval_loader, to_device)
    input_size = train_data.n_characters
    model, opt = get_model(input_size, hidden_size,
                           output_size, embed_size, learning_rate, arch)
    loss_func = nn.NLLLoss()
    train_losses, eval_losses, train_accuracies, eval_accuracies = fit(
        n_epochs, model, loss_func, opt, train_loader, eval_loader, save_every)

    plot_metrics(train_losses, eval_losses, train_accuracies, eval_accuracies)
    torch.save(model, OUT_DIR / 'model.pt')
    save_metrics([train_losses, eval_losses,
                  train_accuracies, eval_accuracies])


def read_config(config_fn):
    with open(config_fn) as f:
        config = json.load(f)
    return config


def set_config(config):
    # hidden_size = 512
    # n_epochs = 15
    # save_every = 50
    # # If you set this too high, it might explode. If too low, it might not learn
    # learning_rate = 0.005
    # batch_size = 16
    # output_size = 2
    # arch = "lstm"
    # OUT_DIR = Path("../processed_data/model/character_rnn/lstm/run_01/")

    global DEVICE, OUT_DIR, DATA_DIR
    DEVICE = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    OUT_DIR = Path(config["out_dir"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR = Path(config["data_dir"])

    hidden_size = config["hidden_size"]
    n_epochs = config["n_epochs"]
    save_every = config["save_every"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    output_size = config["output_size"]
    arch = config["arch"]
    embed_size = config["embed_size"] if ("embed_size" in config) else 16
    train_sets = config["train_sets"]
    eval_sets = config["eval_sets"]

    for k, v in config.items():
        sys.stderr.write(f"{k}={v}\n")

    return hidden_size, n_epochs, save_every, learning_rate, batch_size, output_size, embed_size, train_sets, eval_sets, arch


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def to_device(*args):
    container = []
    for x in args:
        container.append(x.to(DEVICE) if hasattr(x, "to") else x)
    return container


def get_model(input_size, hidden_size, output_size, embed_size, learning_rate, arch):
    if arch == "lstm_embed":
        model = EmbedRNN(input_size, hidden_size,
                         output_size, embed_size).to(DEVICE)
    else:
        model = RNN(input_size, hidden_size, output_size, arch).to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9)
    return model, optimizer


def get_data(batch_size, train_sets, eval_sets, arch):
    # data_dir = Path("../processed_data/preprocess/bioc/propose_on_bioc/")
    data_dir = DATA_DIR
    if arch == "lstm_embed":
        sf_eval = SFData([data_dir / x for x in eval_sets], one_hot=False)
        sf_train = SFData([data_dir / x for x in train_sets],
                          exclude=set(sf_eval.data["seq"]), one_hot=False)
    else:
        sf_eval = SFData([data_dir / x for x in eval_sets], one_hot=True)
        sf_train = SFData([data_dir / x for x in train_sets],
                          exclude=set(sf_eval.data["seq"]), one_hot=True)

    return sf_train, DataLoader(sf_train, batch_size=batch_size, shuffle=True,
                                collate_fn=sf_train._pad_seq), \
        sf_eval, DataLoader(sf_eval, batch_size=batch_size * 4,
                            collate_fn=sf_eval._pad_seq)


def loss_batch(model, loss_func, tensors, seq_lens, labels, opt=None):
    output = model(tensors, seq_lens)
    pred = torch.argmax(output, dim=1)

    loss = loss_func(output, labels)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), pred


def fit(n_epochs, model, loss_func, opt, train_loader, eval_loader, save_every=1000):
    n_steps, train_loss, train_corrects, n_train_examples = 0, 0, 0, 0
    train_losses, train_accuracies, eval_losses, eval_accuracies = [], [], [], []
    start = time.time()

    for epoch in range(n_epochs):
        for tensors, labels, seq_lens, seqs in train_loader:

            model.train()
            n_steps += 1
            loss, pred = loss_batch(
                model, loss_func, tensors, seq_lens, labels, opt)
            train_loss += loss
            train_corrects += sum(pred == labels)
            n_train_examples += len(labels)

            if n_steps % save_every == 0:

                avg_train_loss = train_loss / save_every
                train_losses.append(avg_train_loss)
                train_accuracy = train_corrects / n_train_examples
                train_accuracies.append(train_accuracy)
                train_loss, train_corrects, n_train_examples = 0, 0, 0

                model.eval()
                eval_loss, eval_corrects, n_eval_examples = 0, 0, 0
                with torch.no_grad():
                    for tensors, labels, seq_lens, seqs in eval_loader:
                        loss, pred = loss_batch(
                            model, loss_func, tensors, seq_lens, labels)
                        eval_loss += loss
                        eval_corrects += sum(pred == labels)
                        n_eval_examples += len(labels)
                avg_eval_loss = eval_loss / len(eval_loader)
                eval_accuracy = eval_corrects / n_eval_examples
                eval_losses.append(avg_eval_loss)
                eval_accuracies.append(eval_accuracy)

                print('STEP %d (%s) TRAIN=%.4f ACC=%.4f; EVAL=%.4f ACC=%.4f' %
                      (n_steps, timeSince(start), avg_train_loss, train_accuracy, avg_eval_loss, eval_accuracy))

    return train_losses, eval_losses, train_accuracies, eval_accuracies


def plot_metrics(train_losses, eval_losses, train_accuracies, eval_accuracies):
    eval_losses_smooth = savgol_filter(eval_losses, 15, 4)
    eval_accuracies_smooth = savgol_filter(eval_accuracies, 15, 4)

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


if __name__ == "__main__":
    main()
