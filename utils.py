import os
import pandas as pd
import re
from collections import defaultdict
import atexit


def format_answer(text):
    return text.replace("( ", "(").replace(" )", ")").\
        replace("[ ", "[").replace(" ]", "]")


def extract_examples(df, mode="train"):
    assert mode in ["train", "eval"], "mode must be train or eval."
    contexts = []
    questions = []
    answers = []
    sf = []
    pmid = []
    typ = []
    sent_no = []
    for i, row in df.iterrows():
        try:
            lf = format_answer(row["lf"])
            sentence = row["sent"]
            if (lf in sentence) or mode == "eval":
                contexts.append(sentence)
                questions.append("What does %s stand for?" % row["sf"])
                sf.append(row["sf"])
                answers.append({"text": lf, "score": row["score"]})
                pmid.append(row["pmid"])
                typ.append(row["type"])
                sent_no.append(row["sent_no"])
        except:
            print(f"Sentence {i} is skipped.")
    return contexts, questions, answers, sf, pmid, typ, sent_no


def create_dir_by_fn(fn):
    out_dir = os.path.abspath(os.path.dirname(fn))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


def fasta2table(f, container=None):
    if container == None:
        container = defaultdict(list)

    if hasattr(f, "read"):  # input is a file handle
        pass
    elif isinstance(f, str):
        f = open(f, "r")
        atexit.register(lambda x: x.close(), f)
    else:
        raise ValueError("Input format should be a file handle or a string.")

    for line in f:
        try:
            if re.match(">[0-9]+\|[at]\|[0-9]+", line):
                split_line = line.strip().replace(">", "").split("|")
                pmid = int(split_line[0])
                typ = split_line[1]
                sent_no = int(split_line[2])

            elif line.startswith("  "):
                split_line = line.strip().split("|")
                container["sf"].append(split_line[0])
                container["lf"].append(split_line[1])
                container["score"].append(float(split_line[2]))
                if len(split_line) == 4:
                    container["comment"].append(split_line[3])
                container["pmid"].append(pmid)
                container["type"].append(typ)
                container["sent_no"].append(sent_no)
                container["sent"].append(sent)

            else:
                sent = line.strip()

        except:
            print("error")
            print(line)

    df = pd.DataFrame(container)
    df = df[~df.duplicated()]
    return df


def rerank2table(f, container=None):
    if container == None:
        container = defaultdict(list)

    if hasattr(f, "read"):  # input is a file handle
        pass
    elif isinstance(f, str):
        f = open(f, "r")
        atexit.register(lambda x: x.close(), f)
    else:
        raise ValueError("Input format should be a file handle or a string.")

    for line in f:

        try:
            if line.startswith("input:"): 
                split_line = line.replace("input:", "").strip().replace(">", "").split("|")
                pmid = int(split_line[0])
                typ = split_line[1]
                sent_no = int(split_line[2])

            elif line.startswith("best:"):
                split_line = line.strip().split("\t")
                score = float(split_line[2])
                split_line = split_line[3].split("|")
                container["sf"].append(split_line[0])
                container["lf"].append(split_line[1])
                container["score"].append(score)
                if len(split_line) == 4:
                    container["comment"].append(split_line[3])
                container["pmid"].append(pmid)
                container["type"].append(typ)
                container["sent_no"].append(sent_no)
                container["sent"].append(sent)

            elif line.startswith("sentence:"):
                sent = line.replace("sentence:","").strip()

            else:
                pass

        except:
            print("error")
            print(line)

    df = pd.DataFrame(container)
    df = df[~df.duplicated()]
    return df
