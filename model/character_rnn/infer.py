import click
import sys
sys.path.insert(0, "./model/character_rnn/")
from train import get_data
from data import SFData
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

@click.command()
@click.option("--model_fn", type=str, help="Path to model.")
@click.option("--eval_fn", type=str, help="Path to eval file.")
@click.option("--arch", type=str, help="Model architecture.")
def main(model_fn, eval_fn, arch):
    model = torch.load(model_fn).to(torch.device("cpu"))
    model.eval()

    if arch == "lstm_embed":
        eval_data = SFData([eval_fn], one_hot=False)
    else:
        eval_data = SFData([eval_fn], one_hot=True)
    eval_loader = DataLoader(eval_data, batch_size = 64, collate_fn=eval_data._pad_seq)
    with torch.no_grad():
        container = defaultdict(list)
        for tensors, labels, seq_lens, seqs in eval_loader:
            prob = model(tensors, seq_lens)
            container["prob"] += prob[:, 1].exp().tolist()
            container["pred"] += torch.argmax(prob, dim=1).tolist()
            container["label"] += labels.tolist()
            container["seq"] += list(seqs)

    sys.stdout.write("seq\tprob\tpred\tlabel\n")
    for prob, pred, label, seq in zip(container["prob"], container["pred"], container["label"], container["seq"]):
        sys.stdout.write(f"{seq}\t{prob:.04f}\t{pred}\t{label}\n")

if __name__ == "__main__":
    main()

