#!/usr/bin/env python
# Propose short forms from input text
import sys
import regex as re


def find(text):
    # The regex matches nested parenthesis and brackets
    matches = re.findall(
        " ([\(\[](?>[^\(\)\[\]]+|(?1))*[\)\]])", text)
    for match in matches:
        # Remove enclosing parenthesis
        match = match[1:-1]

        # recurse to find abbreviation within a match:
        find(match)

        # extract abbreviation before "," or ";"
        match = re.split("[,;] ", match)[0]
        sys.stdout.write(f"  {match}|none|-1|todo\n")


def main():
    for line in sys.stdin:
        if line.startswith(">"):  # header line
            sys.stdout.write(line)

        elif line.startswith("  "):  # answer line
            pass

        else:  # context line
            sys.stdout.write(line)

            line = line.strip()
            find(line)


if __name__ == "__main__":
    main()
