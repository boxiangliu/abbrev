import stanza
import os
import click


@click.command()
@click.option("--in_fn", type=str, help="Input file.")
@click.option("--out_fn", type=str, help="Output file.")
def main():
    nlp = stanza.Pipeline("en", processors="tokenize", tokenize_batch_size=64)

    with open(in_fn) as fin, open(out_fn, "w") as fout:
        for i, line in enumerate(fin):
            if i % 1000 == 0:
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




if __name__ == "__main__":
    main()