import pandas as pd
from transformers import BertForQuestionAnswering
from transformers import BertTokenizerFast


nrows = 10
ab3p_fn = "../processed_data/preprocess/ab3p/summarize_ab3p/ab3p_res.csv"
ab3p = pd.read_csv(ab3p_fn, sep="\t", nrows=nrows)

train_contexts = ab3p["sent"].tolist()
train_questions = [f"What does {x} stand for?" for x in ab3p["sf"]]
train_answers = [{"text": x} for x in ab3p["lf"]]


def add_answer_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        text = answer["text"]
        start_idx = context.find(text)
        end_idx = start_idx + len(text)
        answer["answer_start"] = start_idx
        answer["answer_end"] = end_idx

add_answer_idx(train_answers, train_contexts)


model = BertForQuestionAnswering.from_pretrained(
    'bert-large-cased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizerFast.from_pretrained(
    "bert-large-cased-whole-word-masking-finetuned-squad")

train_encodings = tokenizer(train_questions, train_contexts, truncation=True, padding=True)

def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]["answer_start"]))
        end_positions.append(encodings.char_to_token(i, answers[i]["answer_end"]))
        # If None, the answer passage has been truncated. 
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length
    encodings.update({"start_positions": start_positions, "end_positions": end_positions})

add_token_positions(train_encodings, train_answers)

import torch
class Ab3PDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, index):
        return {key: torch.tensor(val[idx]) for key, val in encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


train_dataset = Ab3PDataset(train_encodings)

device = torch.device