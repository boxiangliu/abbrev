import pandas as pd
from transformers import BertForQuestionAnswering
from transformers import BertTokenizerFast
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import os
from transformers import Trainer, TrainingArguments

out_dir = "../processed_data/model/finetune_on_ab3p/"
os.makedirs(out_dir, exist_ok=True)


class Ab3PDataset(torch.utils.data.Dataset):

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def format_answer(text):
    return text.replace("( ", "(").replace(" )", ")").\
        replace("[ ", "[").replace(" ]", "]")


def extract_examples(ab3p):
    contexts = []
    questions = []
    answers = []

    for i, row in ab3p.iterrows():
        lf = format_answer(row["lf"])
        sentence = row["sent"]
        if lf in sentence:
            contexts.append(sentence)
            questions.append("What does %s stand for?" % row["sf"])
            answers.append({"text": lf})
    return contexts, questions, answers


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




nrows = 1e5
ab3p_fn = "../processed_data/preprocess/ab3p/summarize_ab3p/ab3p_res.csv"
ab3p = pd.read_csv(ab3p_fn, sep="\t", nrows=nrows)

model = BertForQuestionAnswering.from_pretrained(
    'bert-large-cased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizerFast.from_pretrained(
    "bert-large-cased-whole-word-masking-finetuned-squad")


contexts, questions, answers = extract_examples(ab3p)

train_pct=0.8
train_div = round(len(contexts) * train_pct)
train_idx = range(train_div)
val_idx = range(train_div, len(contexts))
assert len(train_idx) + len(val_idx) == len(contexts)

train_contexts = contexts[:train_div]
train_questions = questions[:train_div]
train_answers = answers[:train_div]

val_contexts = contexts[train_div:]
val_questions = questions[train_div:]
val_answers = answers[train_div:]

add_answer_idx(train_answers, train_contexts)
add_answer_idx(val_answers, val_contexts)

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

