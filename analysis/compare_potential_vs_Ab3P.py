import sys


def main():
    containers = init_containers()
    for line in sys.stdin:
        if line.startswith("annotation:") or line.startswith("ab3p:"):
            add_annotation_to_containers(line, containers)
        else:
            potential_pairs, Ab3P_pairs = get_potential_and_Ab3P_pairs(
                containers)
            missing_pairs = subtract_list1_from_list2(
                list1=potential_pairs, list2=Ab3P_pairs)
            missing_pairs = classify_missing_pairs(
                missing_pairs, potential_pairs)
            write_SF_and_LF_pairs(missing_pairs)
            containers = init_containers()
            sys.stdout.write(line)


def init_containers():
    return {"PSF": [], "PLF": [], "SF": [], "LF": [], "Ab3P_SF": [], "Ab3P_LF": []}


def add_annotation_to_containers(line, containers):
    line_type = line.strip().split("\t")[0]
    if line_type == "ab3p:":
        entry_type, SF, LF = parse_Ab3P(line)
    elif line_type == "annotation:":
        entry_type, text, position = parse_annotation(line)

    if entry_type == "Ab3P":
        containers["Ab3P_SF"].append({"text": SF, "position": ""})
        containers["Ab3P_LF"].append({"text": LF, "position": ""})
    elif entry_type.startswith("PSF"):
        containers["PSF"].append({"text": text, "position": position})
    elif entry_type.startswith("PLF"):
        containers["PLF"].append({"text": text, "position": position})
    elif entry_type.startswith("SF"):
        containers["SF"].append({"text": text, "position": position})
    elif entry_type.startswith("LF"):
        containers["LF"].append({"text": text, "position": position})


def parse_Ab3P(line):
    split_line = line.strip().split("\t")
    SF, LF, score = split_line[1].split("|")
    return "Ab3P", SF, LF


def parse_annotation(line):
    split_line = line.strip().split("\t")
    if len(split_line) == 4:
        entry_type, text, position = split_line[1:]
    elif len(split_line) == 3:
        entry_type, text = split_line[1:]
        position = ""
    elif len(split_line) == 2:
        entry_type = split_line[1]
        text, position = "", ""
    else:
        raise ValueError

    return entry_type, text, position


def get_potential_and_Ab3P_pairs(container):
    assert len(container["Ab3P_SF"]) == len(container["Ab3P_LF"])
    Ab3P = [{"SF": SF, "LF": LF}
            for SF, LF in zip(container["Ab3P_SF"], container["Ab3P_LF"])]
    assert len(container["PSF"]) == len(container["PLF"])
    potential = [{"SF": PSF, "LF": PLF}
                 for PSF, PLF in zip(container["PSF"], container["PLF"])]
    return potential, Ab3P


def subtract_list1_from_list2(list1, list2):
    missing = []
    for g in list2:
        found = 0
        for p in list1:
            if g["SF"]["text"] == p["SF"]["text"] and (p["LF"]["text"].endswith(g["LF"]["text"]) or p["LF"]["text"].startswith(g["LF"]["text"])):
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

    return missing_pairs


def write_SF_and_LF_pairs(pairs):
    for pair in pairs:
        SF = pair["SF"]["text"]
        LF = pair["LF"]["text"]
        SF_position = pair["SF"]["position"]
        LF_position = pair["LF"]["position"]
        reason = pair["reason"]
        sys.stdout.write(f"annotation:\tSF\t{reason}\t{SF}\t{SF_position}\n")
        sys.stdout.write(f"annotation:\tLF\t{reason}\t{LF}\t{LF_position}\n")


if __name__ == '__main__':
    main()
