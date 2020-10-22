from transformers import pipeline
import click
import torch
from utils import extract_examples

model = "/mnt/scratch/boxiang/projects/abbrev/processed_data/model/finetune_on_ab3p/checkpoint-14500/"
tokenizer = "bert-large-cased-whole-word-masking-finetuned-squad"

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



@click.command()
@click.option("--model", type=str, help="Path to custom model or name of Huggingface built-in model.")
@click.option("--tokenizer", type=str, help="Name of Huggingface built-in tokenizer.")
@click.option("--data_fn", type=str, help="Path to data tsv file.")
def main(model, tokenizer, data_fn):
    device = 0 if torch.cuda.is_available() else -1

    predictor = pipeline("question-answering", 
        model=model, 
        tokenizer=tokenizer,
        device=device)


n=16
timeit predictions = predictor(question=val_questions[:n], context=val_contexts[:n], topk=5)

[x["answer"] for x in predictions]
[x["text"] for x in val_answers[:n]]

