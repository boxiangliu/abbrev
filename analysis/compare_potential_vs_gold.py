import sys


def main():
    containers = {"PSF": [], "PLF": [], "SF": [], "LF": []}
    for line in sys.stdin:
        if line.startswith("annotation:"):
            add_annotation_to_containers(line, containers)
        else:
            potential_pairs, gold_pairs = get_potential_and_gold_pairs(
                containers)
            missing_pairs = subtract_potential_from_gold(
                potential_pairs, gold_pairs)
            missing_pairs = classify_missing_pairs(
                missing_pairs, potential_pairs)
            write_SF_and_LF_pairs(missing_pairs)
            containers = {"PSF": [], "PLF": [], "SF": [], "LF": []}
            sys.stdout.write(line)


def add_annotation_to_containers(line, containers):
    split_line = line.strip().split("\t")

    if len(split_line) == 4:
        type_, text, position = split_line[1:]
    elif len(split_line) == 3:
        type_, text = split_line[1:]
        position = ""
    elif len(split_line) == 2:
        type_ = split_line[1]
        text, position = "", ""
    else:
        raise ValueError

    if type_.startswith("PSF"):
        containers["PSF"].append({"text": text, "position": position})
    elif type_.startswith("PLF"):
        containers["PLF"].append({"text": text, "position": position})
    elif type_.startswith("SF"):
        containers["SF"].append({"text": text, "position": position})
    elif type_.startswith("LF"):
        containers["LF"].append({"text": text, "position": position})


def get_potential_and_gold_pairs(container):
    assert len(container["SF"]) == len(container["LF"])
    gold = [{"SF": SF, "LF": LF}
            for SF, LF in zip(container["SF"], container["LF"])]
    assert len(container["PSF"]) == len(container["PLF"])
    potential = [{"SF": PSF, "LF": PLF}
                 for PSF, PLF in zip(container["PSF"], container["PLF"])]
    return potential, gold


def subtract_potential_from_gold(potential, gold):
    missing = []
    for g in gold:
        found = 0
        for p in potential:
            if g["SF"]["text"] == p["SF"]["text"] and p["LF"]["text"].endswith(g["LF"]["text"]):
                found = 1
        if not found:
            missing.append(g)
    return missing


def classify_missing_pairs(missing_pairs, potential_pairs):
    potential_SFs = [p["SF"]["text"] for p in potential_pairs]
    potential_LFs = [p["LF"]["text"] for p in potential_pairs]

    for m in missing_pairs:
        if m["SF"]["text"] not in potential_SFs:
            m["reason"] = "missing SF"
        elif "|" in m["LF"]["position"]:
            m["reason"] = "multi-span LF"
        elif any([m["LF"]["text"] in LF for LF in potential_LFs]):
            m["reason"] = "gap"
        else:
            m["reason"] = "other"


def write_SF_and_LF_pairs(pairs):
    for pair in pairs:
        SF = pair["SF"]["text"]
        LF = pair["LF"]["text"]
        reason = pair["reason"]
        sys.stdout.write(f"annotation:\tSF\t{reason}\t{SF}\n")
        sys.stdout.write(f"annotation:\tLF\t{reason}\t{LF}\n")


if __name__ == '__main__':
    main()
