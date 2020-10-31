import sys

for line in sys.stdin:
    if line.startswith(">"):
        header = line
        new_line = True
    elif line.startswith("  "):
        answer = line
        if new_line:
            sys.stdout.write(header)
            sys.stdout.write(sentence)
            new_line = False
        sys.stdout.write(answer)
    else:
        sentence = line