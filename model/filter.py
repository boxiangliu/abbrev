import sys
import re
for line in sys.stdin:
    if line.startswith("  "):
        sf = line.strip().split("|")[0]
        match = re.match("[-+]?([0-9]*\.[0-9]+|[0-9]+)[ %]?", sf)
        if match is not None:
            sys.stdout.write(sf + "/n")