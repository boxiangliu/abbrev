from transformers import pipeline
import click
import torch
import sys
sys.path.append(".")
from utils import extract_examples, create_dir_by_fn
import os
import pandas as pd


@click.command()
@click.option("--model", type=str, help="Path to custom model or name of Huggingface built-in model.")
@click.option("--tokenizer", type=str, help="Name of Huggingface built-in tokenizer.")
@click.option("--data_fn", type=str, help="Path to data tsv file.")
@click.option("--out_fn", type=str, help="Path to output file.")
@click.option("--topk", type=int, help="Output top K predictions.", default=5)
@click.option("--nonredundant", is_flag=True, help="Only output non-redundant predictions.")
def main(model, tokenizer, data_fn, out_fn, topk, nonredundant):
    device = 0 if torch.cuda.is_available() else -1

    predictor = pipeline("question-answering",
                         model=model,
                         tokenizer=tokenizer,
                         device=device)

    data = pd.read_csv(data_fn, sep="\t")
    contexts, questions, answers, sfs, pmids, types, sent_nos = extract_examples(
        data, mode="eval")

    if nonredundant:
        # Get more and select the non-redundant topK.
        beam_size = topk * 10
        predictions = predictor(
            question=questions, context=contexts, topk=beam_size)
        assert len(predictions) == beam_size * len(sfs)

    else:
        predictions = predictor(
            question=questions, context=contexts, topk=topk)
        assert len(predictions) == topk * len(sfs)

    create_dir_by_fn(out_fn)

    with open(out_fn, "w") as fout:
        for i, sf in enumerate(sfs):
            context = contexts[i]
            lf = answers[i]["text"]
            score = answers[i]["score"]
            pmid = pmids[i]
            typ = types[i]
            sent_no = sent_nos[i]
            fout.write(f">{pmid}|{typ}|{sent_no}\n")
            fout.write(f"{context}\n")
            fout.write(f"  {sf}|{lf}|{score}|input\n")

            if nonredundant:
                selected_predictions = set()
                n_selected = 0
                j = 0
                while n_selected < topk:
                    prediction = predictions[i * beam_size + j]

                    if prediction not in selected_predictions:

                        selected_predictions.add(prediction)
                        n_selected += 1
                        pred_text = prediction["answer"]
                        pred_score = prediction["score"]
                        fout.write(f"  {sf}|{pred_text}|{pred_score}|bqaf{n_selected+1}|orig{j+1}\n")

                    j += 1
            else:
                for j in range(topk):
                    prediction = predictions[i * topk + j]
                    pred_text = prediction["answer"]
                    pred_score = prediction["score"]
                    fout.write(f"  {sf}|{pred_text}|{pred_score}|bqaf{j+1}\n")


if __name__ == "__main__":
    main()
