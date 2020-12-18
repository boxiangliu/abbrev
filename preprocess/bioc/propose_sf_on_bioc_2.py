#!/usr/bin/env python
# Propose short forms from input text
import sys
import regex as re
import stanza


def main():
    nlp = stanza.Pipeline(lang="en", processors="tokenize")

    for line in sys.stdin:
        sys.stdout.write(line)

        if line.startswith("text:"):
            line = line.strip().split("\t")[1]
            sentences = nlp(line).sentences
            PSFs, PLFs = [], []

            for sentence in sentences:
                sentence = cast_Sentence_to_Span(sentence)
                PSFs, PLFs = extract_PSFs_and_PLFs(sentence, PSFs, PLFs)
                PSFs = strip_fluff(PSFs)
            write_PSFs_and_PLFs(PSFs, PLFs)


def cast_Sentence_to_Span(sentence):
    start = sentence.tokens[0].start_char
    end = sentence.tokens[-1].end_char
    return Span(sentence.text, start, end)


def write_PSFs_and_PLFs(PSFs, PLFs):
    for i, (PSF, PLF) in enumerate(zip(PSFs, PLFs)):
        sys.stdout.write(f"annotation:\tPSF{i}\t{PSF}\n")
        sys.stdout.write(f"annotation:\tPLF{i}\t{PLF}\n")


def extract_PSFs_and_PLFs(sentence, PSFs, PLFs):
    spans = extract_parenthesized_spans(sentence, [])
    for span in spans:

        if span.n_segments == 2:
            PSF, PLF = assign_segments_as_PSF_and_PLF(span.segments)
            PSFs.append(PSF)
            PLFs.append(PLF)

        segment = span.segments[0]

        if n_words(segment) <= 3:
            PLF = span.extract_PLF_before_span()
            PSFs.append(segment)
            PLFs.append(PLF)

        if n_words(segment) >= 2:
            PSF = span.extract_PSF_before_span()
            PSFs.append(PSF)
            PLFs.append(segment)

    return PSFs, PLFs


def assign_segments_as_PSF_and_PLF(segments):
    PSF, PLF = (segments[0], segments[1]) if \
        len(segments[0]) <= len(segments[1]) else (segments[1], segments[0])
    return PSF, PLF


def extract_parenthesized_spans(span, container):
    # The regex matches nested parenthesis and brackets
    matches = re.finditer(
        "([\(\[](?>[^\(\)\[\]]+|(?1))*[\)\]])", span.text)
    for match in matches:
        bounds = [span.start_char + i for i in match.span()]
        match = Span(text=match.group()[1:-1],
                     start_char=bounds[0] + 1,
                     end_char=bounds[1] - 1,
                     parent=span)
        container.append(match)
        # recurse to find abbreviation within a match:
        container = extract_parenthesized_spans(match, container)
    return container


class Span:
    '''A Span instance contains 
    1) the text of the span
    2) the start, end, and length of the span
    3) the enclosing (parent) span
    '''

    def __init__(self, text, start_char, end_char, parent=None):
        self.text = text
        self.start_char = start_char
        self.end_char = end_char
        self.length = end_char - start_char
        self.parent = parent
        self.segments = self.get_segments()
        self.n_segments = len(self.segments)

    def __repr__(self):
        return self.text

    def get_segments(self):
        return re.split("[;,] ", self.text)

    def extract_PLF_before_span(self):
        prefix_length = self.start_char - self.parent.start_char
        prefix = self.parent.text[:prefix_length]
        PLF = prefix.rstrip(" ([")
        return PLF

    def extract_PSF_before_span(self):
        prefix_length = self.start_char - self.parent.start_char
        prefix = self.parent.text[:prefix_length]
        prefix = prefix.rstrip(" ([")
        PSF = prefix.split(" ")[-1]
        return PSF


def n_words(text):
    return len(text.split(" "))


def strip_fluff(PSFs):
    new_PSFs = []
    for PSF in PSFs:
        PSF = strip_whitespaces_and_quotes(PSF)
        PSF = strip_boilerplate_text(PSF)
        new_PSFs.append(PSF)
    return new_PSFs


def strip_whitespaces_and_quotes(text):
    return text.strip(" \'\"")


def strip_boilerplate_text(text):
    return text.replace("for ", "").replace("termed ", "").replace("designated ", "")


if __name__ == "__main__":
    main()
