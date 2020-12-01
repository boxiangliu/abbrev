import click
import sys
sys.path.insert(0, "./model/qa_reject/")
from train import get_data
from data import SFLFData
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
        for sf_tensors, lf_tensors, sf_labels, pair_labels, \
                is_golds, sf_lens, lf_lens, sfs, lfs in eval_loader:
            prob = model(tensors, seq_lens)

            sf_prob, pair_prob = model(sf_tensors, lf_tensors, sf_lens, lf_lens)
            sf_pred, pair_pred = torch.argmax(sf_prob, dim=1), \
                torch.argmax(pair_prob, dim=1)

            container["sf_prob"] += sf_prob[:, 1].exp().tolist()
            container["pair_prob"] += pair_prob[:, 1].exp().tolist()

            container["sf_pred"] += sf_pred.tolist()
            container["pair_pred"] += pair_pred.tolist()

            container["sf_label"] += sf_labels.tolist()
            container["pair_label"] += pair_labels.tolist()

            container["sf"] += list(sfs)
            container["lf"] += list(lfs)

    sys.stdout.write("sf\tsf_prob\tsf_pred\tsf_label\tlf\tpr_prob\tpr_pred\tpr_label\n")
    for sf_prob, pair_prob, sf_pred, pair_pred, sf_label, pair_label, sf, lf in \
        zip(container["sf_prob"], container["pair_prob"], container["sf_pred"], \
            container["pair_pred"], container["sf_label"], container["pair_label"], \
            container["sf"], container["lf"]):
        sys.stdout.write(f"{sf}\t{sf_prob:.04f}\t{sf_pred}\t{sf_label}\t{lf}\t{pair_prob:.04f}\t{pair_pred}\t{pair_label}\n")

if __name__ == "__main__":
    main()

