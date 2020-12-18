import sys


def main():
    containers = {"PSF": [], "PLF": [], "SF": [], "LF": []}
    for line in sys.stdin:
        if line.startswith("annotation:"):
            add_annotation_to_containers(line, containers)
        else:
            sys.stdout.write(line)
            potential, gold = get_potential_and_gold_pairs(containers)
            missing = subtract_potential_from_gold(potential, gold)
            write_SF_and_LF(missing)
            containers = {"PSF": [], "PLF": [], "SF": [], "LF": []}



def add_annotation_to_containers(line, containers):
    type_, text = line.strip().split("\t")[1:3]
    if type_.startswith("PSF"):
        containers["PSF"].append(text)
    elif type_.startswith("PLF"):
        containers["PLF"].append(text)
    elif type_.startswith("SF"):
        containers["SF"].append(text)
    elif type_.startswith("LF"):
        containers["LF"].append(text)


def get_potential_and_gold_pairs(container):
    assert len(container["SF"]) == len(container["LF"])
    gold = [(SF, LF) for SF, LF in zip(container["SF"], container["LF"])]
    assert len(container["PSF"]) == len(container["PLF"])
    potential = [(PSF, PLF)
                 for PSF, PLF in zip(container["PSF"], container["PLF"])]
    return potential, gold


def subtract_potential_from_gold(potential, gold):
    missing = []
    for g in gold:
        found = 0
        for p in potential:
            if g[0] == p[0] and p[1].endswith(g[1]):
                found = 1
        if not found:
            missing.append(g)
    return missing


def write_SFs_and_LFs(SFs, LFs):
    for i, (SF, LF) in enumerate(zip(SFs, LFs)):
        sys.stdout.write(f"annotation:\tSF{i}\t{SF}\n")
        sys.stdout.write(f"annotation:\tLF{i}\t{LF}\n")


if __name__ == '__main__':
    main()
