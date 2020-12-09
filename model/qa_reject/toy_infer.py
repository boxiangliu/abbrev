from pathlib import Path
import sys
sys.path.insert(0, "./model/qa_reject/")
from data import ToyData, WrappedDataLoader
from model import ToyEmbedRNN, ToyEmbedRNNSequence
from torch.utils.data import DataLoader

model_fn = "../processed_data/model/qa_reject/lstm/toy_02/model.pt"
eval_fn = Path("../processed_data/model/qa_reject/toy_data/")
model = torch.load(model_fn).to(torch.device("cpu"))
model.device = torch.device("cpu")
model.eval()


eval_data = ToyData(eval_fn / "test")
eval_loader = DataLoader(eval_data, batch_size = 64, collate_fn=eval_data._pad_seq)
sf_tensors, lf_tensors, labels, sf_lens, lf_lens, sfs, lfs = next(iter(eval_loader))

prob = model(sf_tensors, lf_tensors, sf_lens, lf_lens)
torch.argmax(prob, dim=1)

with torch.no_grad():
    container = defaultdict(list)
    for sf_tensors, lf_tensors, labels, sf_lens, lf_lens, sfs, lfs in eval_loader:

        sf_prob, pair_prob = model(sf_tensors, lf_tensors, sf_lens, lf_lens)
