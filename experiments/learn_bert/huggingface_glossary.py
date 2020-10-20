from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
sequence = "A Titan RTX has 24Gb of VRAM."

tokenized_sequence = tokenizer.tokenize(sequence)
print(tokenized_sequence)
# ['A', 'Titan', 'R', '##T', '##X', 'has', '24', '##G', '##b', 'of', 'V', '##RA', '##M', '.']

inputs = tokenizer.encode(sequence)
print(inputs)
# [101, 138, 18696, 155, 1942, 3190, 1144, 1572, 2349, 1830, 1104, 159, 9664, 2107, 119, 102]

inputs2 = tokenizer(sequence)
print(inputs2)
# {'input_ids': 
# [101, 138, 18696, 155, 1942, 3190, 1144, 1572, 2349, 1830, 1104, 159, 9664, 2107, 119, 102], 
# 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
# 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}


encoded_sequence = inputs2["input_ids"]
decoded_sequence = tokenizer.decode(encoded_sequence)
print(decoded_sequence)
# [CLS] A Titan RTX has 24Gb of VRAM. [SEP]


# Attention masks:
sequence_a = "This is a short sequence."
sequence_b = "This is a long sequence. At least longer than the first one."

tokenized_sequence = tokenizer([sequence_a, sequence_b], padding=True)
print(tokenized_sequence["input_ids"])
print(tokenized_sequence["attention_mask"])


# Sentence pairs
sequence_a1 = sequence_a
sequence_a2 = "What is the type of this sentence?"

sequence_b1 = sequence_b
sequence_b2 = "What is the type of this sentence?"

tokenized_sequence = tokenizer(sequence_a1, sequence_a2)
print(tokenized_sequence)
# {'input_ids': [101, 1188, 1110, 170, 1603, 4954, 119, 102, 1327, 1110, 1103, 2076, 1104, 1142, 5650, 119, 102], 
# 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
# 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

decoded_sequence = tokenizer.decode(tokenized_sequence["input_ids"])
print(decoded_sequence)
# [CLS] This is a short sequence. [SEP] What is the type of this sentence. [SEP]


tokenized_sequence = tokenizer([sequence_a1, sequence_b1], [sequence_a2, sequence_b2])
print(tokenized_sequence)
# {'input_ids': [[101, 1188, 1110, 170, 1603, 4954, 119, 102, 1327, 1110, 1103, 2076, 1104, 1142, 5650, 119, 102], [101, 1188, 1110, 170, 1263, 4954, 119, 1335, 1655, 2039, 1190, 1103, 1148, 1141, 119, 102, 1327, 1110, 1103, 2076, 1104, 1142, 5650, 119, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}

decoded_sequence = tokenizer.decode(tokenized_sequence["input_ids"][0])
print(decoded_sequence)
# [CLS] This is a short sequence. [SEP] What is the type of this sentence. [SEP]
# tokenizer.decode() can only take one sequence and not a list of sequences. 

 





