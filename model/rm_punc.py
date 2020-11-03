#!/user/bin/env python3
import re
import string
for line in sys.stdin:
    if line.startswith("best:"):
        split_line = line.split("\t")
        answer = split_line[3]
        split_answer = answer.split("|")
        lf = split_answer[1]
        if re.match(f"[{string.punctuaction}]", lf):
            lf = re.sub(f"^[{string.punctuaction}]+", "", lf)
            split_answer[1] = lf
            answer = "|".join(split_answer)
            split_line[3] = answer
            line = "\t".join(split_line)
        sys.stdout.write(line)
    else:
        sys.stdout.write(line)

