import sys
import re

for line in sys.stdin:
    split_line = line.split("\t")
    sf = split_line[0]
    freq = split_line[1]
    ec_count = dict()

    ec = ""
    for c in sf:
        if re.match("[0-9]", c):
            ec += "0"
        elif re.match("[A-Z]", c):
            ec += "A"
        elif re.match("[a-z]", c):
            ec += "a"
        else:
            ec += "-"
    ec = re.sub("AAAA+", "AAA...", ec)
    ec = re.sub("aaaa+", "aaa...", ec)
    ec = re.sub("0000+", "000...", ec)
    ec = re.sub("----+", "---...", ec)
    out_line = [sf, ec, freq]