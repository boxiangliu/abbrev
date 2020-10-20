squad_train_fn = "../data/SQuAD2.0/train-v2.0.json"

import json

with open(squad_train_fn) as fin:
    squad = json.load(fin) # dictionary format

squad = squad["data"] # list format
article = squad[0] # dict format
title = article["title"] # string format
paragraphs = article["paragraphs"] # list format
paragraph = paragraphs[0] # dictionary format
qas = paragraph["qas"] # list format
context = paragraph["context"] # string format
for i in qas:
    print(i)
