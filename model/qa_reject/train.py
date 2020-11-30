import torch
import torch.nn as nn
import sys
sys.path.append("./model/qa_reject/")
from data import SFLFData, WrappedDataLoader
from torch.utils.data import DataLoader
from model import EmbedRNN
import time
import math
import pickle
from pathlib import Path
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import json
import click

# config_fn = "../processed_data/model/qa_reject/lstm/run_01/config.json"


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
    train_losses, train_sf_losses, train_pair_losses, eval_losses, eval_sf_losses, eval_pair_losses, \
        train_sf_accuracies, train_pair_accuracies, eval_sf_accuracies, eval_pair_accuracies = fit(
            n_epochs, model, loss_func, opt, train_loader, eval_loader, save_every)

    plot_metrics(train_losses, train_sf_losses, train_pair_losses, eval_losses, eval_sf_losses, eval_pair_losses,
                 train_sf_accuracies, train_pair_accuracies, eval_sf_accuracies, eval_pair_accuracies)
    torch.save(model, OUT_DIR / 'model.pt')
    save_metrics([train_losses, train_sf_losses, train_pair_losses, eval_losses, eval_sf_losses, eval_pair_losses,
                  train_sf_accuracies, train_pair_accuracies, eval_sf_accuracies, eval_pair_accuracies])


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
    model = EmbedRNN(input_size, hidden_size,
                     output_size, embed_size).to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9)
    return model, optimizer


def get_data(batch_size, arch):
    data_dir = DATA_DIR
    if arch == "lstm_embed":
        sf_eval = SFLFData([data_dir / "medstract"], one_hot=False)
        sf_train = SFLFData([data_dir / "Ab3P", data_dir / "bioadi",
                             data_dir / "SH"], exclude=set(sf_eval.sf_lf_pairs), one_hot=False)
    else:
        sf_eval = SFLFData([data_dir / "medstract"], one_hot=True)
        sf_train = SFLFData([data_dir / "Ab3P", data_dir / "bioadi",
                             data_dir / "SH"], exclude=set(sf_eval.sf_lf_pairs), one_hot=True)

    return sf_train, DataLoader(sf_train, batch_size=batch_size, shuffle=True,
                                collate_fn=sf_train._pad_seq), \
        sf_eval, DataLoader(sf_eval, batch_size=batch_size * 4,
                            collate_fn=sf_eval._pad_seq)


def loss_batch(model, loss_func, sf_tensors, lf_tensors, sf_lens, lf_lens, sf_labels, pair_labels, opt=None):
    sf_output, pair_output = model(sf_tensors, lf_tensors, sf_lens, lf_lens)
    sf_pred, pair_pred = torch.argmax(
        sf_output, dim=1), torch.argmax(pair_output, dim=1)

    sf_loss = loss_func(sf_output, sf_labels)
    pair_loss = loss_func(pair_output, pair_labels)
    loss = sf_loss + pair_loss

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), sf_loss.item(), pair_loss.item(), sf_pred, pair_pred


def eval(model, loss_func, eval_loader, eval_losses, eval_sf_losses, eval_pair_losses, eval_sf_accuracies, eval_pair_accuracies):
    model.eval()
    eval_loss, eval_sf_loss, eval_pair_loss, eval_sf_corrects, eval_pair_corrects, n_eval_examples = [
        0 for _ in range(6)]
    with torch.no_grad():
        for sf_tensors, lf_tensors, sf_labels, pair_labels, \
                is_golds, sf_lens, lf_lens, sfs, lfs in eval_loader:
            loss, sf_loss, pair_loss, sf_pred, pair_pred = loss_batch(
                model, loss_func, sf_tensors, lf_tensors, sf_lens, lf_lens, sf_labels, pair_labels)

            eval_loss += loss
            eval_sf_loss += sf_loss
            eval_pair_loss += pair_loss

            eval_sf_corrects += sum(sf_pred == sf_labels)
            eval_pair_corrects += sum(pair_pred == pair_labels)

            n_eval_examples += len(sf_labels)

    avg_eval_loss, avg_eval_sf_loss, avg_eval_pair_loss = \
        eval_loss / save_every, eval_sf_loss / save_every, eval_pair_loss / save_every
    eval_sf_accuracy, eval_pair_accuracy = \
        eval_sf_corrects / n_eval_examples, eval_pair_corrects / n_eval_examples

    eval_losses.append(avg_eval_loss)
    eval_sf_losses.append(avg_eval_sf_loss)
    eval_pair_losses.append(avg_eval_pair_loss)

    eval_sf_accuracies.append(eval_sf_accuracy)
    eval_pair_accuracies.append(eval_pair_accuracy)

    return avg_eval_loss, avg_eval_sf_loss, avg_eval_pair_loss, eval_sf_accuracy, eval_pair_accuracy, \
        eval_losses, eval_sf_losses, eval_pair_losses, eval_sf_accuracies, eval_pair_accuracies


def fit(n_epochs, model, loss_func, opt, train_loader, eval_loader, save_every=1000):
    n_steps, train_loss, train_sf_loss, train_pair_loss, train_sf_corrects, train_pair_corrects, n_train_examples = [
        0 for _ in range(7)]
    train_losses, train_sf_losses, train_pair_losses, train_sf_accuracies, train_pair_accuracies, \
        eval_losses, eval_sf_losses, eval_pair_losses, eval_sf_accuracies, eval_pair_accuracies = [
            [] for _ in range(10)]
    start = time.time()

    for epoch in range(n_epochs):
        for sf_tensors, lf_tensors, sf_labels, pair_labels, \
                is_golds, sf_lens, lf_lens, sfs, lfs in train_loader:

            model.train()
            n_steps += 1
            loss, sf_loss, pair_loss, sf_pred, pair_pred = \
                loss_batch(model, loss_func, sf_tensors, lf_tensors,
                           sf_lens, lf_lens, sf_labels, pair_labels, opt)

            train_loss += loss
            train_sf_loss += sf_loss
            train_pair_loss += pair_loss

            train_sf_corrects += sum(sf_pred == sf_labels)
            train_pair_corrects += sum(pair_pred == pair_labels)

            n_train_examples += len(sf_labels)

            if n_steps % save_every == 0:

                avg_train_loss, avg_train_sf_loss, avg_train_pair_loss = \
                    train_loss / save_every, train_sf_loss / \
                    save_every, train_pair_loss / save_every

                train_losses.append(avg_train_loss)
                train_sf_losses.append(avg_train_sf_loss)
                train_pair_losses.append(avg_train_pair_loss)

                train_sf_accuracy, train_pair_accuracy = \
                    train_sf_corrects / n_train_examples, train_pair_corrects / n_train_examples

                train_sf_accuracies.append(train_sf_accuracy)
                train_pair_accuracies.append(train_pair_accuracy)

                train_loss, train_sf_loss, train_pair_loss, train_sf_corrects, train_pair_corrects, n_train_examples = [
                    0 for _ in range(6)]

                avg_eval_loss, avg_eval_sf_loss, avg_eval_pair_loss, eval_sf_accuracy, eval_pair_accuracy, \
                    eval_losses, eval_sf_losses, eval_pair_losses, eval_sf_accuracies, eval_pair_accuracies = \
                    eval(model, loss_func, eval_loader, eval_losses, eval_sf_losses,
                         eval_pair_losses, eval_sf_accuracies, eval_pair_accuracies)

                print('STEP %d (%s) TRAIN: L=%.4f SF_L=%.4f PAIR_L=%.4f SF_ACC=%.4f PAIR_ACC=%.4f; EVAL: L=%.4f SF_L=%.4f PAIR_L=%.4f SF_ACC=%.4f PAIR_ACC=%.4f' %
                      (n_steps, timeSince(start), avg_train_loss, avg_train_sf_loss, avg_train_pair_loss, train_sf_accuracy, train_pair_accuracy,
                       avg_eval_loss, avg_eval_sf_loss, avg_eval_pair_loss, eval_sf_accuracy, eval_pair_accuracy))

    return train_losses, train_sf_losses, train_pair_losses, eval_losses, eval_sf_losses, eval_pair_losses, \
        train_sf_accuracies, train_pair_accuracies, eval_sf_accuracies, eval_pair_accuracies


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


def plot_metrics(train_losses, train_sf_losses, train_pair_losses, eval_losses, eval_sf_losses, eval_pair_losses,
                 train_sf_accuracies, train_pair_accuracies, eval_sf_accuracies, eval_pair_accuracies):
    # start from here. turn the following code into a for loop.
    train_metrics = [train_losses, train_sf_losses,
                     train_pair_losses, train_sf_accuracies, train_pair_accuracies]
    eval_metrics = [eval_losses, eval_sf_losses,
                    eval_pair_losses, eval_sf_accuracies, eval_pair_accuracies]
    criterions = ["loss"] * 3 + ["accuracy"] * 2
    types = ["total", "SF", "pair", "SF", "pair"]
    for train_metric, eval_metric, criterion, _type in zip(train_metrics, eval_metrics, criterions, types):
        legend_position = "upper right" if _type == "loss" else "lower right"
        title, ylabel, out_fn = f"{criterion} {_type}", _type, OUT_DIR / f"{criterion}_{_type}.png"
        plot_single_metric(train_metric, eval_metric,
                           legend_position, title, ylable, out_fn)


def save_metrics(metrics):
    with open(OUT_DIR / "all_losses.pkl", "wb") as fout:
        pickle.dump(metrics, fout)


if __name__ == "__main__":
    main()
