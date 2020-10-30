proposal_fn = ""
gold_fn = ""

import sys
sys.path.append(".")
from utils import fasta2table


from collections import defaultdict
proposal = fasta2table(proposal_fn)
gold = fasta2table(gold_fn)

