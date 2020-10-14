import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

model = BertForQuestionAnswering.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad")

question = "How many parameters does BERT-large have?"
answer_text = "BERT-large is really big… it has 24-layers and an embedding size of 1,024, for a total of 340M parameters! Altogether it is 1.34GB, so expect it to take a couple minutes to download to your Colab instance."
input_ids = tokenizer.encode(question, answer_text)
tokens = tokenizer.convert_ids_to_tokens(input_ids)


# Search the input_ids for the first instance of the `[SEP]` token.
sep_index = input_ids.index(tokenizer.sep_token_id)

# The number of segment A tokens includes the [SEP] token istelf.
num_seg_a = sep_index + 1

# The remainder are segment B.
num_seg_b = len(input_ids) - num_seg_a

# Construct the list of 0s and 1s.
segment_ids = [0] * num_seg_a + [1] * num_seg_b

# There should be a segment_id for every input token.
assert len(segment_ids) == len(input_ids)

start_scores, end_scores = model(torch.tensor([input_ids]),  # The tokens representing our input text.
                                 token_type_ids=torch.tensor([segment_ids]))  # The segment IDs to differentiate question from answer_text

answer_start = torch.argmax(start_scores)
answer_end = torch.argmax(end_scores)

answer = " ".join(tokens[answer_start:answer_end+1])

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

print('Answer: "' + answer + '"')


bert_abstract = "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial taskspecific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement)."
question = "What does BERT mean?"


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


answer_question(question, bert_abstract)



answer_text = "We introduce an approach for simultaneous horizontal and vertical integration, Linked Matrix Factorization (LMF), for the general case where some matrices share rows (e.g., features) and some share columns (e.g., samples)."
question = "What is LMF?"
question = "What is features?"
answer_question(question, answer_text)


answer_text = df.iloc[0]["sent"]
answer_text
question = "What is %s?" % (df.iloc[0]["sf"])
answer_question(question, answer_text)
# Under otherwise unchanged conditions the reference substance ifosfamide (IF) -- a further development of cyclophosphamide
# answer: a

answer_text = df.iloc[1]["sent"]
print(answer_text)
question = "What is %s?" % (df.iloc[1]["sf"])
print(question)
answer_question(question, answer_text)


i = 2
answer_text = df.iloc[i]["sent"]
print(answer_text)
question = "What is %s?" % (df.iloc[i]["sf"])
print(question)
answer_question(question, answer_text)

i = 3
answer_text = df.iloc[i]["sent"]
print(answer_text)
question = "What is %s?" % (df.iloc[i]["sf"])
print(question)
answer_question(question, answer_text)


for i in range(100):
    answer_text = df.iloc[i]["sent"]
    sf = df.iloc[i]["sf"]
    lf = df.iloc[i]["lf"]
    question = "What is %s?" % sf
    answer = answer_question(question, answer_text)
    print(answer_text)
    print(f"{sf}\t{answer}\t{lf}")


# Both wrong:
The apparent isoelectric points (pI) in isoelectric focusing (IF) of human pituitary and amniotic fluid prolactin (hPRL), both non-iodinated and iodinated, were determined.
pI      apparent isoelectric points     points

