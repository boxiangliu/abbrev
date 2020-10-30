import sys
sys.path.append(".")
from utils import fasta2table

proposal_fn = "../processed_data/model/propose/MED1250_proposal"
gold_fn = "../processed_data/preprocess/med1250/fltr_answerable/MED1250_labeled"


proposal = fasta2table(proposal_fn)
gold = fasta2table(gold_fn)


proposal
