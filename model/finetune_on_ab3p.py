import pandas as pd
from transformers import BertForQuestionAnswering
from transformers import BertTokenizerFast
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import os
from transformers import Trainer, TrainingArguments
from utils import extract_examples


out_dir = "../processed_data/model/finetune_on_ab3p/"
os.makedirs(out_dir, exist_ok=True)


class Ab3PDataset(torch.utils.data.Dataset):

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)



def add_answer_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        text = answer["text"]
        start_idx = context.find(text)
        end_idx = start_idx + len(text)
        answer["answer_start"] = start_idx
        answer["answer_end"] = end_idx


def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(
            i, answers[i]["answer_start"]))
        end_positions.append(encodings.char_to_token(
            i, answers[i]["answer_end"]))
        # If None, the answer passage has been truncated.
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length
    encodings.update({"start_positions": start_positions,
                      "end_positions": end_positions})




train_fn = "../processed_data/preprocess/model/train_val/train.tsv"
val_fn = "../processed_data/preprocess/model/train_val/val.tsv"
train_data = pd.read_csv(train_fn, sep="\t")
val_data = pd.read_csv(val_fn, sep="\t")


train_contexts, train_questions, train_answers, _ = extract_examples(train_data)
val_contexts, val_questions, val_answers, _ = extract_examples(val_data)

add_answer_idx(train_answers, train_contexts)
add_answer_idx(val_answers, val_contexts)

model = BertForQuestionAnswering.from_pretrained(
    'bert-large-cased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizerFast.from_pretrained(
    "bert-large-cased-whole-word-masking-finetuned-squad")

train_encodings = tokenizer(
    train_questions, train_contexts, truncation=True, padding=True)
val_encodings = tokenizer(val_questions, val_contexts,
                          truncation=True, padding=True)

add_token_positions(train_encodings, train_answers)
add_token_positions(val_encodings, val_answers)

train_dataset = Ab3PDataset(train_encodings)
val_dataset = Ab3PDataset(val_encodings)


training_args = TrainingArguments(
    output_dir=out_dir,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=out_dir,
    logging_steps=10)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset)

trainer.train()

