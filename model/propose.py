import sys
import regex as re


def main():
    for line in sys.stdin:
        if line.startswith(">"):  # header line
            sys.stdout.write(line)

        elif line.startswith("  "):  # answer line
            pass

        else:  # context line
            sys.stdout.write(line)
            # The regex matches nested parenthesis and brackets
            matches = re.findall(
                "[\(\[](?>[^\(\)\[\]]+|(?R))*[\)\]]", line.strip())
            for match in matches:
                match = match[1:-1]

                # extract abbreviation before "," or ";"
                match = re.split("[,;] ", match)[0]
                sys.stdout.write(f"  {match}|none|-1|todo\n")

if __name__ == "__main__":
    main()
