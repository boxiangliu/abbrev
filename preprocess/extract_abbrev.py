import re
import os
import click
import pycountry


def get_countries():
    countries = [x.name for x in list(pycountry.countries)]
    return countries


@click.command()
@click.option("--in_fn", type=str, help="Input file.")
@click.option("--out_fn", type=str, help="Output file.")
def main(in_fn, out_fn):
    countries = get_countries()

    with open(in_fn) as fin, open(out_fn, "w") as fout:
        for line in fin:
            split_line = line.strip().split("|")
            pmid = split_line[0]
            typ = split_line[1]
            line_no = split_line[2]
            text = split_line[3]

            for m in re.finditer(" \((.*?)\)[ ,\.\?:\'\"]", text):
                m_text = m.group(1)
                start = m.start(1)
                end = m.end(1)

                abb_type = 0 
                if re.search("\s", m_text): # space
                    abb_type = 1
                elif not re.search("[a-zA-Z]", m_text): # no letter
                    abb_type = 2
                elif re.match("^(IX|IV|V?I{0,3})$", m_text.upper()): # Roman numeral
                    abb_type = 3
                elif re.match("^[a-z]$", m_text): # only lowercase letter
                    abb_type = 4
                elif re.search("[pP][<>=≤≥⩽⩾]", m_text): # p-value 
                    abb_type = 5
                elif re.search("[nN]=", m_text): # sample size
                    abb_type = 6
                elif re.search("[=±]", m_text): # definition and unit
                    abb_type = 7
                elif re.search("°C", m_text): # temperature
                    abb_type = 8
                elif re.search("(\.com|\.gov)", m_text): # URLs
                    abb_type = 9
                elif m_text in countries:
                    abb_type = 10

                out = f"{pmid}|{typ}|{line_no}|{start},{end}|{abb_type}|{m_text}|{text}\n"
                fout.write(out)


if __name__ == "__main__":
    main()