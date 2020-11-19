import click
import sys
sys.path.append("./model/character_rnn/")
from train import get_data
from data import SFData
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

@click.command()
@click.option("--model_fn", type=str, help="Path to model.")
@click.option("--eval_fn", type=str, help="Path to eval file.")
def main(model_fn, eval_fn):
    model = torch.load(model_fn)
    model.eval()

    eval_data = SFData([eval_fn])
    eval_loader = DataLoader(eval_data, batch_size = 64, collate_fn=eval_data.pack_seq)
    with torch.no_grad():
        container = defaultdict(list)
        for tensors, labels, seq_lens, seqs in eval_loader:
            container["pred"] += torch.argmax(model(tensors), dim=1).tolist()
            container["label"] += labels.tolist()
            container["seq"] += seqs.tolist()

    for pred, label, seq in zip(container["pred"], container["label"], container["seq"]):
        sys.stdout.write(f"{seq}\t{pred}\t{label}\n")

