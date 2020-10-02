import re
import os
import click


@click.command()
@click.option("--in_fn", type=str, help="Input file.")
@click.option("--out_fn", type=str, help="Output file.")
def main(in_fn, out_fn):
    with open(in_fn) as fin, open(out_fn, "w") as fout:
        for line in fin:
            split_line = line.strip().split("|")
            pmid = split_line[0]
            typ = split_line[1]
            line_no = split_line[2]
            text = split_line[3]

            for m in re.finditer("\((.*?)\)", text):
                m_text = m.group(1)
                start = m.start(1)
                end = m.end(1)
                out = f"{pmid}|{typ}|{line_no}|{start},{end}|{m_text}|{text}\n"
                fout.write(out)


if __name__ == "__main__":
    main()