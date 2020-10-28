import os
import pandas as pd
import re

def format_answer(text):
    return text.replace("( ", "(").replace(" )", ")").\
        replace("[ ", "[").replace(" ]", "]")


def extract_examples(ab3p):
    contexts = []
    questions = []
    answers = []
    sf = []
    pmid = []
    typ = []
    sent_no = [] 
    for i, row in ab3p.iterrows():
        try:
            lf = format_answer(row["lf"])
            sentence = row["sent"]
            if lf in sentence:
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


def fasta2table(f, container):
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

