import sys
import regex as re

for line in sys.stdin:
    if line.startswith("  "):
        sf = line.strip().split("|")[0]

        fltr = False
        m1 = re.search("[-+]?([0-9]*\.[0-9]+|[0-9]+)[ %]", sf)
        if m1 is not None:
            fltr = True

        m2 = re.match("[0-9]+", sf)
        if m2 is not None:
            fltr = True

        m3 = re.match("^(IX|IV|V?I{0,3})$", sf.upper())
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

        m7 = re.search("[a-g]$", sf)
        if m7 is not None:
            fltr = True

        if fltr == False:
            sys.stdout.write(sf + "\n")
