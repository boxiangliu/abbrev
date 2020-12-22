import sys
import re
from ipdb import set_trace

def main():
    container = init_container()
    write_output_header()
    lines = open("../processed_data/preprocess/bioc/propose_sf_on_bioc_2/Ab3P").readlines()
    for line in lines:
        parse_line(line, container)


def write_output_header():
    sys.stdout.write(f"PSF\tPLF\tPSF_label\tPLF_char_labels\tPLF_word_labels\tpmid\ttype\n")


def init_container():
    return {"PSF": [], "PLF": [], "SF": [], "LF": [], "pmid": "", "type": ""}


def reset_container(container):
    for key in ["PSF", "PLF", "SF", "LF"]:
        container[key] = []


def parse_line(line, container):
    if line.startswith("id:"):
        container["pmid"] = line.strip().split("\t")[1]
    elif line.startswith("annotation:"):
        parse_annotation_line(line, container)
    else:
        if line.startswith("text:"):
            update_text_type(container)
        write_BIO_data(container)
        reset_container(container)


def parse_annotation_line(line, container):
    split_line = line.strip().split("\t")
    if len(split_line) >= 3:
        entry_type, text = split_line[1:3]
    elif len(split_line) == 2:
        entry_type, text = split_line[1], ""

    entry_type = entry_type.strip("0123456789")
    text = text.strip()
    container[entry_type].append(text)


def update_text_type(container):
    if container["type"] == "":
        container["type"] = "title"
    elif container["type"] == "title":
        container["type"] = "abstract"
    elif container["type"] == "abstract":
        container["type"] = "title"


def check_container(container):
    assert len(container["PLF"]) == len(container["PSF"])
    assert len(container["SF"]) == len(container["LF"])


def write_BIO_data(container):
    check_container(container)

    for PSF, PLF in zip(container["PSF"], container["PLF"]):
        SF_match = 0
        for SF, LF in zip(container["SF"], container["LF"]):
            if SF == PSF:
                SF_match = 1
                if LF in PLF:
                    write_positive_BIO_instance(
                        PSF, PLF, LF, container["pmid"], container["type"])
                else:
                    write_negative_BIO_instance(
                        PSF, PLF, 1, container["pmid"], container["type"])
        if not SF_match:
            write_negative_BIO_instance(
                PSF, PLF, 0, container["pmid"], container["type"])


def write_positive_BIO_instance(PSF, PLF, LF, pmid, text_type):
    try:
        start_chars = [m.start() for m in re.finditer(f"[^ ]{LF}[$ ]", PLF)]
    except:
        set_trace()
    length = len(LF)
    PLF_char_labels = make_PLF_char_labels(PLF, start_chars, length)
    PLF_word_labels = char_to_word_labels(PLF, PLF_char_labels)
    PLF_char_labels = ",".join(PLF_char_labels)
    PLF_word_labels = ",".join(PLF_word_labels)
    PSF_label = 1
    sys.stdout.write(f"{PSF}\t{PLF}\t{PSF_label}\t{PLF_char_labels}\t{PLF_word_labels}\t{pmid}\t{text_type}\n")


def make_PLF_char_labels(PLF, start_chars, length):
    PLF_char_labels = ["O" for _ in PLF]
    for start_char in start_chars:
        PLF_char_labels[start_char:start_char+length] = ["B"] + ["I"] * (length - 1)
    return PLF_char_labels


def char_to_word_labels(text, char_labels):
    word_labels = [char_labels[0]]
    for i, char in enumerate(text):
        if char == " ":
            word_labels.append(char_labels[i + 1])
    return word_labels


def write_negative_BIO_instance(PSF, PLF, PSF_label, pmid, text_type):
    if PSF != "" and PLF != "":
        PLF_char_labels = ",".join(["O" for char in PLF])
        PLF_word_labels = ",".join(["O" for word in PLF.split(" ")])
        sys.stdout.write(f"{PSF}\t{PLF}\t{PSF_label}\t{PLF_char_labels}\t{PLF_word_labels}\t{pmid}\t{text_type}\n")


if __name__ == "__main__":
    main()
