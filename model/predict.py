from transformers import pipeline
import click
import torch
import sys
sys.path.append(".")
from utils import extract_examples
import os

# model = "/mnt/scratch/boxiang/projects/abbrev/processed_data/model/finetune_on_ab3p/checkpoint-14500/"
# tokenizer = "bert-large-cased-whole-word-masking-finetuned-squad"

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
@click.option("--out_fn", type=str, help="Path to output file.")
@click.option("--topk", type=int, help="Output top K predictions.", default=5)
def main(model, tokenizer, data_fn, topk):
    device = 0 if torch.cuda.is_available() else -1

    predictor = pipeline("question-answering", 
        model=model, 
        tokenizer=tokenizer,
        device=device)

    contexts, questions, answers, sfs = extract_examples(ab3p)

    predictions = predictor(question=questions, context=contexts, topk=topk)
    assert len(predictions) == topk * len(sfs)

    if not os.path.exists(os.path.dirname(out_fn)): 
        os.makedirs(os.path.dirname(out_fn))

    with open(out_fn, "w") as fout:
        for i, sf in enumerate(sfs):
            context = contexts[i]
            lf = answers[i]["text"]
            score = answers[i]["score"]
            fout.write(f"{context}\n")
            fout.write(f"  ab3p|{sf}|{lf}|{score}\n")
            for j in range(topk):
                prediction = prediction[i * topk + j]
                pred_text = prediction["text"]
                pred_score = prediction["score"]
                fout.write(f"  bqaf{j+1}|{sf}|{pred_text}|{pred_score}\n")


if __name__ == "__main__":
    main()
