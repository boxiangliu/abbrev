import stanza
import sys
from tqdm import tqdm

def get_token_interval(token):
    start, end = token.misc.split("|")
    start = int(start.replace("start_char=", ""))
    end = int(end.replace("end_char=", ""))
    return start, end


def subset(interval1, interval2):
    return (interval1[0] >= interval2[0]) and (interval1[1] <= interval2[1])


nlp = stanza.Pipeline(lang="en", processors="tokenize")
text_len, text_counter = 0, 0
sfs, lfs, intervals = [], [], {}

for line in tqdm(sys.stdin):
    if line.startswith("id:"):
        pmid = int(line.strip().split("\t")[1])

    elif line.startswith("text:"):
        if sfs != []:
            for sent_no, sentence in enumerate(text.sentences):
                for token in sentence.tokens:
                    token_interval = get_token_interval(token)
                    prefix = f"{pmid}\t{text_type}\t{sent_no}"
                    suffix = ""
                    for interval, form_type in intervals.items():
                        if subset(token_interval, interval):
                            suffix = f"B-{form_type}" \
                                if token_interval[0] == interval[0] else f"I-{form_type}"
                    if suffix == "":
                        suffix = "O"
                    sys.stdout.write(f"{prefix}\t{token.text}\t{suffix}\n")
        sfs, lfs, intervals = [], [], {}


        text = line.strip().split("\t")[1]
        prev_len = text_len
        text_len = len(text)
        text = nlp(text)

        text_counter += 1
        if text_counter == 1:
            text_type = "title"
            text_offset = 0

        elif text_counter == 2:
            text_type = "abstract"
            text_offset = prev_len + 1
            text_counter = 0

    elif line.startswith("annotation:"):
        split_line = line.strip().split("\t")
        form_type = split_line[1]
        form_text = split_line[2]
        form_interval = [[int(y) for y in x.split("+")] for x in split_line[3].split("|")]

        if form_type.startswith("SF"):
            sfs.append(form_text)
            for start, length in form_interval:
                intervals[(start - text_offset, start + length - text_offset)] = "SF"

        elif form_type.startswith("LF"):
            lfs.append(form_text)
            for start, length in form_interval:
                intervals[(start - text_offset, start + length - text_offset)] = "LF"
