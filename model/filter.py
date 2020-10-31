import sys
import re
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

        m4 = re.match("[A-Za-z]", sf)
        if m4 is None:
            sys.stdout.write(sf + "\n")