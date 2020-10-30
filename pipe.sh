##############
# Preprocess #
##############

# Extract abstract from PubMed XML file:
bash preprocess/extract_abstract.sh


# Split sentences:
bash preprocess/split_sent.sh


# Extract abbreviations: 
bash preprocess/extract_abbrev.sh


# Count the occurrances of each abbreviation:
python3 preprocess/count_abbrev.py


# Use Ab3P to extract abbreviations:
bash preprocess/ab3p/run_ab3p.sh --in_dir /mnt/scratch/boxiang/projects/abbrev/processed_data/preprocess/sentence/ --out_dir /mnt/scratch/boxiang/projects/abbrev/processed_data/preprocess/ab3p/run_ab3p/
python3 preprocess/ab3p/summarize_ab3p.py


# Reformat MED1250 to FASTA format:
python3 preprocess/med1250/text2fasta.py --med1250_fn "../data/MED1250/MED1250_labeled" --out_fn "../processed_data/preprocess/med1250/text2fasta/MED1250_labeled"

# Extract answerable sentences. 
# An answerable sentence is one with a short form and long form pair.
python3 preprocess/med1250/fltr_answerable.py

# Get the unlabeled dataset:
grep -v "^  " ../processed_data/preprocess/med1250/fltr_answerable/MED1250_labeled > ../processed_data/preprocess/med1250/fltr_answeralbe/MED1250_unlabeled

# Convert labeled dataset to tsv format:
python preprocess/med1250/fasta2table.py --in_fn ../processed_data/preprocess/med1250/fltr_answeralbe/MED1250_labeled --out_fn ../processed_data/preprocess/med1250/fltr_answeralbe/MED1250_labeled.tsv

#########
# Model #
#########
# Create training data:
python3 model/make_data.py --nrows 1e6 --ab3p_fn ../processed_data/preprocess/ab3p/summarize_ab3p/ab3p_res.csv --out_dir ../processed_data/preprocess/model/data_1M/
python3 model/make_data.py --nrows 1000 --ab3p_fn ../processed_data/preprocess/ab3p/summarize_ab3p/ab3p_res.csv --out_dir ../processed_data/preprocess/model/data_1K/


# Finetuning: 
python3 model/finetune_on_ab3p.py


# Predictions:
bash model/predict_run.sh ../processed_data/preprocess/model/data_1M/val.tsv ../processed_data/preprocess/model/predict/ab3p_ft_data_1M/ ../processed_data/model/finetune_on_ab3p/checkpoint-final/
bash model/predict_run.sh ../processed_data/preprocess/model/data_1M/val.tsv ../processed_data/preprocess/model/predict/squad_ft_data_1M/ bert-large-cased-whole-word-masking-finetuned-squad


# Propose short forms: 
cat ../processed_data/preprocess/med1250/fltr_answerable/MED1250_labeled | python model/propose.py > ../processed_data/model/propose/MED1250_proposal

############
# Evaluate #
############
# Run Ab3P on MED1250 data:
bash preprocess/ab3p/run_ab3p.sh --in_fn /mnt/scratch/boxiang/projects/abbrev/processed_data/preprocess/med1250/fltr_answeralbe/MED1250_unlabeled --out_fn /mnt/scratch/boxiang/projects/abbrev/processed_data/evaluate/MED1250/MED1250_ab3p

# Run Ab3P-fine-tuned BERT model on MED1250 data:
python model/predict.py --model ../processed_data/model/finetune_on_ab3p/checkpoint-final/ --tokenizer bert-large-cased-whole-word-masking-finetuned-squad --data_fn ../processed_data/preprocess/med1250/fltr_answeralbe/MED1250_labeled.tsv --out_fn ../processed_data/evaluate/MED1250/MED1250_bert_ab3p_ft

# Run SQuAD-fine-tuned BERT model on MED1250 data: 
python model/predict.py --model bert-large-cased-whole-word-masking-finetuned-squad --tokenizer bert-large-cased-whole-word-masking-finetuned-squad --data_fn ../processed_data/preprocess/med1250/fltr_answeralbe/MED1250_labeled.tsv --out_fn ../processed_data/evaluate/MED1250/MED1250_bert_squad_ft


############
# Analysis #
############
# TBD:
python model/analysis/ab3p5bert.py 


