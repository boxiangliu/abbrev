import sys
import regex as re

for line in sys.stdin:
    if line.startswith("  "):
        sf = line.strip().split("|")[0]

        fltr = False
        m1 = re.search("[-+]?([0-9]*\.[0-9]+|[0-9]+)[ %]", sf)
        if m1 is not None:
            print(1)
            fltr = True

        m2 = re.search("^[0-9]+$", sf)
        if m2 is not None:
            print(2)
            fltr = True

        m3 = re.search("^(IX|IV|V?I{0,3})$", sf.upper())
        if m3 is not None:
            print(3)
            fltr = True

        m4 = re.search("[A-Za-z]", sf)
        if m4 is None:
            print(4)
            fltr = True

        m5 = re.findall(" ([\(\[](?>[^\(\)\[\]]+|(?1))*[\)\]])", sf)
        if len(m5) > 0:
            print(5)
            fltr = True

        m6 = re.search("[<>=≤≥⩽⩾±]", sf)
        if m6 is not None:
            print(6)
            fltr = True

        m7 = re.search("^[a-g]$", sf)
        if m7 is not None:
            print(7)
            fltr = True

        if fltr == True:
            sys.stdout.write(sf + "\n")
