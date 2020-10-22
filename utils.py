import os
import pandas as pd

def format_answer(text):
    return text.replace("( ", "(").replace(" )", ")").\
        replace("[ ", "[").replace(" ]", "]")


def extract_examples(ab3p):
    contexts = []
    questions = []
    answers = []
    sf = []
    for i, row in ab3p.iterrows():
        lf = format_answer(row["lf"])
        sentence = row["sent"]
        if lf in sentence:
            contexts.append(sentence)
            questions.append("What does %s stand for?" % row["sf"])
            sf.append(row["sf"])
            answers.append({"text": lf, "score": row["score"]})
    return contexts, questions, answers, sf


def check_dir_exists(fn):
    out_dir = os.path.abspath(os.path.dirname(fn))
    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)