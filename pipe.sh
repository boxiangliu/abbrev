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
bash preprocess/ab3p/run_ab3p.sh
python3 preprocess/ab3p/summarize_ab3p.py


# Reformat MED1250 to FASTA format:
python3 preprocess/med1250/text2fasta.py --med1250_fn "../data/MED1250/MED1250_labeled" --out_dir "../processed_data/preprocess/med1250/text2fasta/MED1250_labeled"



# Extract answerable sentences. 
# An answerable sentence is one with a short form and long form pair.
python3 preprocess/med1250/fltr_answerable.py


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



############
# Analysis #
############
python model/analysis/ab3p5bert.py 