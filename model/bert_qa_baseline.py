import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import pandas as pd
from collections import defaultdict
import os

model = BertForQuestionAnswering.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad")
out_dir = "../processed_data/model/bert_qa_baseline/"
os.makedirs(out_dir, exist_ok=True)

def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text)

    # Report how long the input sequence is.
    # print('Query has {:,} tokens.\n'.format(len(input_ids)))

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example question through the model.
    start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                    token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
        
        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        
        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    # print('Answer: "' + answer + '"')

    return answer


df = pd.read_csv("../processed_data/preprocess/ab3p/summarize_ab3p/ab3p_res.csv", sep="\t", nrows=1e4)

container = defaultdict(list)
for i in range(df.shape[0]):
    print(f"Question {i}")
    answer_text = df.iloc[i]["sent"]
    sf = df.iloc[i]["sf"]
    lf = df.iloc[i]["lf"]
    question = "What does %s stand for?" % sf
    answer = answer_question(question, answer_text)
    container["text"].append(answer_text)
    container["ab3p"].append(sf)
    container["bert"].append(lf)
    container["answer"].append(answer)


df2 = pd.DataFrame(container)
df2.to_csv(f"{out_dir}/results.tsv", sep="\t", index=False)

num_diff = 0
for i in range(df2.shape[0]):
    ab3p = df2.iloc[i]["ab3p"]
    bert = df2.iloc[i]["bert"]
    question = df2.iloc[i]["text"]
    sf = df2.iloc[i]["sf"]
    if ab3p.replace(" ", "").lower() != bert.replace(" ", "").lower():
        print(f"Question {i}")
        print(question)
        print(f"sf: {sf}")
        print(f"ab3p: {ab3p}")
        print(f"bert: {bert}")
        num_diff += 1

accuracy = (df2.shape[0] - num_diff) / df2.shape[0]
# 82.3%

correct = df2[df2["ab3p"].str.replace(" ", "").str.lower() == df2["bert"].str.replace(" ", "").str.lower()].iloc[:10]
for row in correct.itertuples():
    ab3p = row.ab3p
    bert = row.bert
    question = row.text
    sf = row.sf

    print(question)
    print(f"sf: {sf}")
    print(f"ab3p: {ab3p}")
    print(f"bert: {bert}")


incorrect = df2[df2["ab3p"].str.replace(" ", "").str.lower() != df2["bert"].str.replace(" ", "").str.lower()].iloc[:10]
for row in incorrect.itertuples():
    ab3p = row.ab3p
    bert = row.bert
    question = row.text
    sf = row.sf

    print(question)
    print(f"sf: {sf}")
    print(f"ab3p: {ab3p}")
    print(f"bert: {bert}")
