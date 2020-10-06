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

            for m in re.finditer(" \((.*?)\)[,\.\?:\'\"]", text):
                m_text = m.group(1)
                start = m.start(1)
                end = m.end(1)

                abb_type = 0 
                if " " in m_text: # space
                    abb_type = 1
                elif not re.search("[a-zA-Z]", m_text): # no letter
                    abb_type = 2
                elif re.match("^(IX|IV|V?I{0,3})$", m_text.upper()): # Roman numeral
                    abb_type = 3
                elif re.match("^[a-z]$", m_text): # only lowercase letter
                    abb_type = 4
                elif re.search("[pP]<0", m_text): # p-value 
                    abb_type = 5
                out = f"{pmid}|{typ}|{line_no}|{start},{end}|{abb_type}|{m_text}|{text}\n"
                fout.write(out)


if __name__ == "__main__":
    main()