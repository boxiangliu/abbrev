import stanza
import os

in_fn = "../processed_data/preprocess/abstract/pubmed19n0972.txt"
out_dir = "../processed_data/preprocess/sentence/"
os.makedirs(out_dir, exist_ok=True)
out_fn = os.path.join(out_dir,"pubmed19n0972.txt")

stanza.download("en")
nlp = stanza.Pipeline("en", processors="tokenize", tokenize_batch_size=64)

with open(in_fn) as fin, open(out_fn, "w") as fout:
    for i, line in enumerate(fin):
        if i % 100 == 0:
            print(f"INFO\t{i} lines processed.")
        split_line = line.strip().split("|")
        try:
            pmid = split_line[0]
            typ = split_line[1]
            text = split_line[2]
            doc = nlp(text)
            for j, sentence in enumerate(doc.sentences):
                out = f"{pmid}|{typ}|{j}|{sentence.text}\n"
                fout.write(out)
        except:
            pass



