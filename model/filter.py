import sys
import regex as re

for line in sys.stdin:
    if line.startswith("  "):
        sf = line.strip().split("|")[0]
        m1 = re.match("[-+]?([0-9]*\.[0-9]+|[0-9]+)[ %]", sf)
        if m1 is not None:
            pass
            # sys.stdout.write(sf + "\n")

        m2 = re.match("[0-9]+", sf)
        if m2 is not None:
            pass
            # sys.stdout.write(sf + "\n")


        m3 = re.match("^(IX|IV|V?I{0,3})$", sf.upper())
        if m3 is not None:
            pass
            # sys.stdout.write(sf + "\n")

        m4 = re.search("[A-Za-z]", sf)
        if m4 is None:
            pass
            # sys.stdout.write(sf + "\n")

        m5 = re.findall(" ([\(\[](?>[^\(\)\[\]]+|(?1))*[\)\]])", sf)
        if len(m5) > 0:
            sys.stdout.write(sf + "\n")
