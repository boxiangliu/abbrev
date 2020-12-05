import sys
from collections import OrderedDict
from tqdm import tqdm
"""
Schema for container:
pmid -> text -> sf -> [gold_lf, ab3p_lf, score]
"""
container = OrderedDict()
for line in tqdm(sys.stdin):
    if line.startswith("id:"):
        pmid = line.strip().split("\t")[1]
        container[pmid] = OrderedDict()

    elif line.startswith("text:"):
        text = line.strip().split("\t")[1]
        container[pmid][text] = OrderedDict() 

    elif line.startswith("ab3p:"):
        ab3p_sf, ab3p_lf, score = line.strip().split("\t")[1].split("|")
        container[pmid][text][ab3p_sf] = ["", ab3p_lf, score]

    elif line.startswith("annotation:"):
        form_type, annotation = line.strip().split("\t")[1:3]
        if form_type.startswith("SF"):
            gold_sf = annotation
        elif form_type.startswith("LF"):
            gold_lf = annotation

            if gold_sf in container[pmid][text]:
                container[pmid][text][gold_sf][0] = gold_lf
            else:
                container[pmid][text][gold_sf] = [gold_lf, "", -1]

    """
    if sf and gold_lf == ab3p_lf, then correct_sf = 1; correct_lf = 1; extra = 0.
    if sf and gold_lf != ab3p_lf, then correct_sf = 1; correct_lf = 0; extra = 0.
    if sf and ab3p_lf == None, then correct_sf = 0; correct_lf = 0; extra = 0.
    lf sf and gold_lf == None, then correct_sf = 0; correct_lf = 0; extra = 1. 
    """
sys.stdout.write("pmid\ttext\tsf\tgold_lf\tab3p_lf\tscore\tcorrect_sf\tcorrect_lf\textra_pair\n")
for pmid, pmid_content in container.items():
    for text, text_content in pmid_content.items():
        for sf, (gold_lf, ab3p_lf, score) in text_content.items():
            if gold_lf == ab3p_lf:
                correct_sf, correct_lf, extra_pair = 1, 1, 0
            elif ab3p_lf == "":
                correct_sf, correct_lf, extra_pair = 0, 0, 0
            elif gold_lf == "":
                correct_sf, correct_lf, extra_pair = 0, 0, 1
            elif gold_lf != ab3p_lf:
                correct_sf, correct_lf, extra_pair = 1, 0, 0
            sys.stdout.write(f"{pmid}\t{text}\t{sf}\t{gold_lf}\t{ab3p_lf}\t{score}\t{correct_sf}\t{correct_lf}\t{extra_pair}\n")

