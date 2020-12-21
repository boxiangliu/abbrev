import sys


def main():
    container = init_container()
    write_output_header()
    for line in sys.stdin:
        parse_line(line, container)


def write_output_header():
    sys.stdout.write(f"PSF\tPLF\tPSF_label\tPLF_char_labels\tPLF_word_labels\tpmid\ttype\n")


def init_container():
    return {"PSF": [], "PLF": [], "SF": [], "LF": [], "pmid": "", "type": ""}


def parse_line(line, container):
    if line.startswith("id:"):
        container["pmid"] = line.strip().split("\t")[1]
    elif line.startswith("annotation:"):
        entry_type, text = line.strip().split("\t")[1:3]
        entry_type = entry_type.strip("0123456789")
        text = text.strip()
        container[entry_type].append(text)
    else:
        if line.startswith("text:"):
            update_text_type(container)
        write_BIO_data(container)


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


def pseudo_match(PSF, PLF, SF, LF):
    # TODO: add condition for multi-span LFs
    return PSF == SF and LF in PLF


def write_BIO_data(container):
    check_container(container)

    for PSF, PLF in zip(container["PSF"], container["PLF"]):
        has_pseudo_match = 0
        for SF, LF in zip(container["SF"], container["LF"]):
            if pseudo_match(PSF, PLF, SF, LF):
                has_pseudo_match = 1
                write_positive_BIO_instance(PSF, PLF, SF, LF, container[
                                            "pmid"], container["type"])
        if not has_pseudo_match:
            write_negative_BIO_instance(
                PSF, PLF, container["pmid"], container["type"])


def write_positive_BIO_instance(PSF, PLF, SF, LF, pmid, text_type):
    start_char = PLF.index(LF)
    length = len(LF)
    PLF_char_labels = make_PLF_char_labels(PLF, start_char, length)
    PLF_word_labels = char_to_word_labels(PLF, PLF_char_labels)
    PSF_label = 1
    sys.stdout.write(f"{PSF}\t{PLF}\t{PSF_label}\t{PLF_char_labels}\t{PLF_word_labels}\t{pmid}\t{text_type}\n")


def make_PLF_char_labels(PLF, start_char, length):
    PLF_char_labels = []
    for i, char in enumerate(PLF):
        if i == star_char:
            PLF_char_labels.append("B")
        elif i > star_char and i <= start_char + length:
            PLF_char_labels.append("I")
        else:
            PLF_char_labels.append("O")
    return PLF_char_labels


def char_to_word_labels(text, char_labels):
    word_labels = [char_label[0]]
    for i, char in enumerate(text):
        if char == "":
            word_labels.append(char_labels[i + 1])
    return word_labels


def write_negative_BIO_instance(PSF, PLF, pmid, text_type):
    PLF_char_labels = ["O" for char in PLF]
    PLF_word_labels = ["O" for word in PLF.split(" ")]
    PSF_label = 0
    sys.stdout.write(f"{PSF}\t{PLF}\t{PSF_label}\t{PLF_char_labels}\t{PLF_word_labels}\t{pmid}\t{text_type}\n")


if __name__ == "__main__":
    main()