from pathlib import Path
import sys
sys.path.insert(0, "./model/qa_reject/")
from data import ToyData, WrappedDataLoader
from model import ToyEmbedRNN, ToyEmbedRNNSequence
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

model_fn = "../processed_data/model/qa_reject/lstm/toy_06/model.pt"
eval_fn = Path("../processed_data/model/qa_reject/toy_data3/")
model = torch.load(model_fn).to(torch.device("cpu"))
model.device = torch.device("cpu")
model.eval()


eval_data = ToyData(eval_fn / "test")
eval_loader = DataLoader(eval_data, batch_size = 64, collate_fn=eval_data._pad_seq)
sf_tensors, lf_tensors, labels, sf_lens, lf_lens, sfs, lfs = next(iter(eval_loader))

prob, attn = model(sf_tensors, lf_tensors, sf_lens, lf_lens)
torch.argmax(prob, dim=1)
labels
len((labels != torch.argmax(prob, dim=1)).nonzero())

for i in range(20):
    sfs[i]
    lfs[i]

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attn[:sf_lens[i].item(),i,:lf_lens[i].item()].detach().numpy(), cmap='bone')
    ax.set_xticklabels(" " + lfs[i], Fontsize=6)
    ax.set_yticklabels(" " + sfs[i], Fontsize=6)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(cax)
    plt.savefig(f"test{i}.pdf")

with torch.no_grad():
    container = defaultdict(list)
    for sf_tensors, lf_tensors, labels, sf_lens, lf_lens, sfs, lfs in eval_loader:
        sf_prob, pair_prob = model(sf_tensors, lf_tensors, sf_lens, lf_lens)
