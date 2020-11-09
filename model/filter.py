import sys
import regex as re
import click

def filter(sf):
    fltr = False
    m1 = re.search("[-+]?([0-9]*\.[0-9]+|[0-9]+)[ %]", sf)
    if m1 is not None:
        fltr = True

    m2 = re.search("^[0-9]+$", sf)
    if m2 is not None:
        fltr = True

    m3 = re.search("^(IX|IV|V?I{0,3})$", sf.upper())
    if m3 is not None:
        fltr = True

    m4 = re.search("[A-Za-z]", sf)
    if m4 is None:
        fltr = True

    m5 = re.findall(" ([\(\[](?>[^\(\)\[\]]+|(?1))*[\)\]])", sf)
    if len(m5) > 0:
        fltr = True

    m6 = re.search("[<>=≤≥⩽⩾±]", sf)
    if m6 is not None:
        fltr = True

    m7 = re.search("^[a-g]$", sf)
    if m7 is not None:
        fltr = True

    return fltr


@click.command()
@click.option("--ftype", type=str, help="file type (fasta or table)", default="fasta")
@click.option("--column", type=int, help="If file is table, which column is the SF.", default=3)
def main(ftype, column):
    for line in sys.stdin:
        if ftype == "fasta":
            if line.startswith("  "):
                sf = line.strip().split("|")[0]

                fltr = filter(sf)

                if fltr == False:
                    sys.stdout.write(line)

            else:
                sys.stdout.write(line)


        elif ftype == "table":
            print(line)
            sf = line.strip().split("\t")[column]

            fltr = filter(sf)

            if fltr == False:
                sys.stdout.write(line)



if __name__ == "__main__":
    main()