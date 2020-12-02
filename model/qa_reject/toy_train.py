import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, "./model/qa_reject/")
from data import ToyData, WrappedDataLoader
from model import ToyEmbedRNN
import time
import math
import pickle
from pathlib import Path
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import json
import click

# config_fn = "../processed_data/model/qa_reject/lstm/toy_01/config.json"


@click.command()
@click.option("--config_fn", type=str, help="Path to configuration file.")
def main(config_fn):
    config = read_config(config_fn)
    hidden_size, n_epochs, save_every, learning_rate, \
        batch_size, output_size, embed_size, arch = set_config(config)

    train_data, train_loader, eval_data, eval_loader = get_data(
        batch_size, arch)
    train_loader = WrappedDataLoader(train_loader, to_device)
    eval_loader = WrappedDataLoader(eval_loader, to_device)
    input_size = train_data.n_characters
    model, opt = get_model(input_size, hidden_size,
                           output_size, embed_size, learning_rate, arch)
    loss_func = nn.NLLLoss()
    train_losses, eval_losses, train_accuracies, eval_accuracies = fit(
        n_epochs, model, loss_func, opt, train_loader, eval_loader, save_every)

    torch.save(model, OUT_DIR / 'model.pt')
    save_metrics([train_losses, eval_losses,
                  train_accuracies, eval_accuracies])
    plot_metrics(train_losses, eval_losses, eval_losses,
                 train_accuracies, eval_accuracies)


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

    for k, v in config.items():
        sys.stderr.write(f"{k}={v}\n")

    return hidden_size, n_epochs, save_every, learning_rate, batch_size, output_size, embed_size, arch


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
    model = ToyEmbedRNN(input_size, hidden_size,
                        output_size, embed_size).to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9)
    return model, optimizer


def get_data(batch_size, arch):
    data_dir = DATA_DIR
    eval_data = ToyData(data_dir / "test")
    train_data = ToyData(data_dir / "train")

    return train_data, DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                  collate_fn=train_data._pad_seq), \
        eval_data, DataLoader(eval_data, batch_size=batch_size * 4,
                              collate_fn=eval_data._pad_seq)


def loss_batch(model, loss_func, sf_tensors, lf_tensors, sf_lens, lf_lens, labels, opt=None):
    output = model(sf_tensors, lf_tensors, sf_lens, lf_lens)
    pred = torch.argmax(output, dim=1)

    loss = loss_func(output, labels)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), pred


def eval(model, loss_func, eval_loader, eval_losses, eval_accuracies):
    model.eval()
    eval_loss, eval_corrects, n_eval_examples = [0 for _ in range(3)]
    with torch.no_grad():
        for sf_tensors, lf_tensors, labels, sf_lens, lf_lens, sfs, lfs in eval_loader:
            loss, pred = loss_batch(
                model, loss_func, sf_tensors, lf_tensors, sf_lens, lf_lens, labels)

            eval_loss += loss
            eval_corrects += sum(pred == labels)
            n_eval_examples += len(labels)

    avg_eval_loss = eval_loss / len(eval_loader)
    eval_accuracy = eval_sf_corrects / n_eval_examples

    eval_losses.append(avg_eval_loss)
    eval_accuracies.append(eval_accuracy)

    return avg_eval_loss, eval_accuracy, eval_losses, eval_accuracies


def fit(n_epochs, model, loss_func, opt, train_loader, eval_loader, save_every=1000):
    n_steps, train_loss, train_corrects, n_train_examples = [0 for _ in range(4)]
    train_losses, train_accuracies, eval_losses, eval_accuracies = [[] for _ in range(4)]
    start = time.time()

    for epoch in range(n_epochs):
        for sf_tensors, lf_tensors, labels, sf_lens, lf_lens, sfs, lfs in train_loader:

            model.train()
            n_steps += 1
            loss, pred = \
                loss_batch(model, loss_func, sf_tensors, lf_tensors,
                           sf_lens, lf_lens, labels, opt)

            train_loss += loss
            train_corrects += sum(pred == labels)

            n_train_examples += len(labels)

            if n_steps % save_every == 0:

                avg_train_loss, = train_loss / save_every
                train_losses.append(avg_train_loss)

                train_accuracy = train_sf_corrects / n_train_examples
                train_accuracies.append(train_accuracy)

                train_loss, train_corrects, n_train_examples = [0 for _ in range(3)]

                avg_eval_loss, eval_accuracy, eval_losses, eval_accuracies = \
                    eval(model, loss_func, eval_loader, eval_losses, eval_accuracies)

                print('STEP %d (%s) TRAIN: L=%.4f ACC=%.4f; EVAL: L=%.4f ACC=%.4f' %
                      (n_steps, timeSince(start), avg_train_loss, train_accuracy, avg_eval_loss, eval_accuracy))

    return train_losses, eval_losses, train_accuracies, eval_accuracies


def plot_single_metric(train_metric, eval_metric, legend_position, title, ylabel, out_fn):
    train_metric_smooth = savgol_filter(train_metric, 15, 4)
    eval_metric_smooth = savgol_filter(eval_metric, 15, 4)

    plt.close()
    plt.plot(train_metric, label="train")
    plt.plot(train_metric_smooth, label="smooth train")
    plt.plot(eval_metric, label="eval")
    plt.plot(eval_metric_smooth, label="smooth eval")
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel(ylabel)
    plt.legend(loc=legend_position)
    plt.savefig(out_fn)


def plot_metrics(train_losses, eval_losses, train_accuracies, eval_accuracies):
    # start from here. turn the following code into a for loop.
    train_metrics = [train_losses, train_accuracies]
    eval_metrics = [eval_losses, eval_accuracies]
    criterions = ["loss", "accuracy"]
    for train_metric, eval_metric, criterion in zip(train_metrics, eval_metrics, criterions):
        legend_position = "upper right" if criterion == "loss" else "lower right"
        title, ylabel, out_fn = criterion, criterion, OUT_DIR / f"{criterion}.png"
        plot_single_metric(train_metric, eval_metric,
                           legend_position, title, ylabel, out_fn)


def save_metrics(metrics):
    with open(OUT_DIR / "all_losses.pkl", "wb") as fout:
        pickle.dump(metrics, fout)


if __name__ == "__main__":
    main()
