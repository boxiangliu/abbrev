import click
import sys
sys.path.insert(0, "./model/qa_reject/")
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
    model.device = torch.device("cpu")
    model.eval()
    eval_data = SFLFData([eval_fn], one_hot=False)
    eval_loader = DataLoader(eval_data, batch_size=64,
                             collate_fn=eval_data._pad_seq)
    with torch.no_grad():
        container = defaultdict(list)
        for sf_tensors, lf_tensors, sf_labels, pair_labels, \
                is_golds, sf_lens, lf_lens, sfs, lfs in eval_loader:

            prob, attn_weights = model(
                sf_tensors, lf_tensors, sf_lens, lf_lens)
            pred = torch.argmax(prob, dim=1)

            container["prob"] += prob[:, 1].exp().tolist()
            container["pred"] += pred.tolist()
            container["label"] += pair_labels.tolist()
            container["sf"] += list(sfs)
            container["lf"] += list(lfs)

    sys.stdout.write("sf\tlf\tprob\tpred\tlabel\n")
    for prob, pred, label, sf, lf in \
        zip(container["prob"], container["pred"], container["label"],
            container["sf"], container["lf"]):
        sys.stdout.write(f"{sf}\t{lf}\t{prob:.04f}\t{pred}\t{label}\n")

if __name__ == "__main__":
    main()
