import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler, BatchSampler
import numpy as np

feature = []
label = []
for i in range(100):
    if i < 90:
        feature.append(i)
        label.append(0)
    else:
        feature.append(i)
        label.append(1)

df = pd.DataFrame({"feature": feature, "labels": label})

all_feature = torch.tensor([feature for feature in df["feature"]])
all_label_ids = torch.tensor([label for label in df["labels"]], dtype=torch.long)

train_data = TensorDataset(all_label_ids, all_feature)
BATCH_SIZE = 10

labels_unique, counts = np.unique(df["labels"], return_counts=True)
class_weights = [sum(counts) / c for c in counts]

example_weights = [class_weights[e] for e in df["labels"]]

sampler = WeightedRandomSampler(example_weights, 8)
train_dataloader = DataLoader(train_data, batch_sampler=BatchSampler(sampler, batch_size=10, drop_last=False))


arr_batch = []
for step, batch in enumerate(train_dataloader):
    batch = tuple(t for t in batch)
    label_ids = batch
    arr_batch.append(label_ids)
arr_batch

for i in range(len(arr_batch)):
    print(torch.sum(arr_batch[i]))