from transformers import pipeline
import click
import torch
import sys
sys.path.append(".")
from utils import extract_examples, create_dir_by_fn
import os
import pandas as pd
from collections import OrderedDict

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

    data = pd.read_csv(data_fn, sep="\t", dtype={"pmid":str, "sent_no": str})
    data = data.loc[lambda x: x["comment"] != "omit"]
    contexts, questions, answers, sfs, pmids, types, sent_nos = extract_examples(
        data, mode="eval")

    sys.stderr.write("Getting predictions...")

    predictions = predictor(
        question=questions, context=contexts, topk=topk)
    assert len(predictions) == topk * len(sfs)

    create_dir_by_fn(out_fn)

    sys.stderr.write("Writing to file...")
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
                selected_predictions = OrderedDict()
                n_selected = 0
                for j in range(topk):
                    prediction = predictions[i * topk + j]
                    pred_text = prediction["answer"]
                    pred_score = prediction["score"]

                    if pred_text not in selected_predictions:
                        n_selected += 1
                        selected_predictions[pred_text] = [sf, pred_text, pred_score, f"bqaf{n_selected}", pred_score, f"orig{j+1}"]
                    else:
                        selected_predictions[pred_text][4] += pred_score

                for l in selected_predictions.values():
                    out_line = "|".join([str(x) for x in l])
                    fout.write("  " + out_line + "\n")

            else:
                for j in range(topk):
                    prediction = predictions[i * topk + j]
                    pred_text = prediction["answer"]
                    pred_score = prediction["score"]
                    fout.write(f"  {sf}|{pred_text}|{pred_score}|bqaf{j+1}\n")



if __name__ == "__main__":
    main()
