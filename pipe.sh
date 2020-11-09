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
cat ../processed_data/preprocess/med1250/text2fasta/MED1250_labeled | python3 preprocess/med1250/fltr_answerable.py > ../processed_data/preprocess/med1250/fltr_answerable/MED1250_labeled


# Get the unlabeled dataset:
grep -v "^  " ../processed_data/preprocess/med1250/fltr_answerable/MED1250_labeled > ../processed_data/preprocess/med1250/fltr_answerable/MED1250_unlabeled


# Convert labeled dataset to tsv format:
python preprocess/med1250/fasta2table.py --in_fn ../processed_data/preprocess/med1250/fltr_answerable/MED1250_labeled --out_fn ../processed_data/preprocess/med1250/fltr_answerable/MED1250_labeled.tsv


# Get training PMIDs:
cut -f5 ../processed_data/preprocess/propose/pubmed19n0*.tsv | uniq | python preprocess/trng_PMIDs.py --med1250_pmid ../data/MED1250/MED1250_PMID --biotext_pmid ../data/BioText/BioText_PMID > ../processed_data/preprocess/trng_PMIDs/trng_PMID


#########
# Model #
#########
# Create training data:
python3 model/make_data.py --nrows 1e6 --ab3p_fn ../processed_data/preprocess/ab3p/summarize_ab3p/ab3p_res.csv --out_dir ../processed_data/preprocess/model/data_1M/
python3 model/make_data.py --nrows 1000 --ab3p_fn ../processed_data/preprocess/ab3p/summarize_ab3p/ab3p_res.csv --out_dir ../processed_data/preprocess/model/data_1K/


# Finetuning: 
python3 model/finetune_on_ab3p.py


# Predictions on ab3p-identified short form:
bash model/predict_run.sh ../processed_data/preprocess/model/data_1M/val.tsv ../processed_data/model/predict/ab3p_ft_data_1M/ ../processed_data/model/finetune_on_ab3p/checkpoint-final/
bash model/predict_run.sh ../processed_data/preprocess/model/data_1M/val.tsv ../processed_data/model/predict/squad_ft_data_1M/ bert-large-cased-whole-word-masking-finetuned-squad


# Propose short forms:
cat ../processed_data/preprocess/med1250/fltr_answerable/MED1250_labeled | python model/propose.py > ../processed_data/model/propose/MED1250_proposal
python preprocess/med1250/fasta2table.py --in_fn ../processed_data/model/propose/MED1250_proposal --out_fn ../processed_data/model/propose/MED1250_proposal.tsv
cat ../processed_data/preprocess/med1250/text2fasta/MED1250_labeled | python model/propose.py > ../processed_data/model/propose/MED1250_all_proposal
python preprocess/med1250/fasta2table.py --in_fn ../processed_data/model/propose/MED1250_all_proposal --out_fn ../processed_data/model/propose/MED1250_all_proposal.tsv


cat ../processed_data/preprocess/med1250/fltr_answerable/MED1250_labeled | python model/propose.py | python model/filter.py > ../processed_data/model/propose/MED1250_filtered
python preprocess/med1250/fasta2table.py --in_fn ../processed_data/model/propose/MED1250_filtered --out_fn ../processed_data/model/propose/MED1250_filtered.tsv


# Propose possible short forms:
bash preprocess/propose.sh ../processed_data/preprocess/sentence/ ../processed_data/preprocess/propose/
bash preprocess/fasta2table.sh ../processed_data/preprocess/propose/ ../processed_data/preprocess/propose/


# Predict on proposed short forms:
for fn in `ls ../processed_data/preprocess/propose/pubmed*.tsv`; do
    base=`basename $fn .tsv`; echo $base
    sbatch -p  1080Ti,1080Ti_mlong,1080Ti_slong,2080Ti,2080Ti_mlong,M40x8,M40x8_mlong,M40x8_slong,P100,TitanXx8,TitanXx8_mlong,TitanXx8_slong,V100_DGX,V100x8 \
        --gres=gpu:1 --job-name=$base --output=../processed_data/model/predict/propose/${base}.log \
        --wrap "python model/predict.py --model bert-large-cased-whole-word-masking-finetuned-squad --tokenizer bert-large-cased-whole-word-masking-finetuned-squad \
        --nonredundant --topk 20 --data_fn $fn --out_fn ../processed_data/model/predict/propose/${base}.fasta"
done

# Count (SF, LF, rank) frequencies:
cat ../processed_data/model/predict/propose/pubmed19n0*.fasta | python model/3D_freq.py --topk 5 | sort -k1,1 -k2,2 -k3,3 > ../processed_data/model/3D_freq/3D_freq_sort.tsv


############
# Evaluate #
############
# Run Ab3P on MED1250 data:
bash preprocess/ab3p/run_ab3p.sh --in_fn /mnt/scratch/boxiang/projects/abbrev/processed_data/preprocess/med1250/fltr_answerable/MED1250_unlabeled --out_fn /mnt/scratch/boxiang/projects/abbrev/processed_data/evaluate/MED1250/MED1250_ab3p

# Run Ab3P-fine-tuned BERT model on MED1250 data using gold-standard SF:
python model/predict.py --model ../processed_data/model/finetune_on_ab3p/checkpoint-final/ --tokenizer bert-large-cased-whole-word-masking-finetuned-squad --data_fn ../processed_data/preprocess/med1250/fltr_answerable/MED1250_labeled.tsv --out_fn ../processed_data/evaluate/MED1250/MED1250_bert_ab3p_ft

# Run SQuAD-fine-tuned BERT model on MED1250 data using gold-standard SF: 
python model/predict.py --model bert-large-cased-whole-word-masking-finetuned-squad --tokenizer bert-large-cased-whole-word-masking-finetuned-squad --data_fn ../processed_data/preprocess/med1250/fltr_answerable/MED1250_labeled.tsv --out_fn ../processed_data/evaluate/MED1250/MED1250_bert_squad_ft


# Evaluate short form proposal:
python evaluate/proposal5gold.py


# Run Ab3P-fine-tuned BERT model on MED1250 data using proposed SF:
python model/predict.py --model ../processed_data/model/finetune_on_ab3p/checkpoint-final/ --tokenizer bert-large-cased-whole-word-masking-finetuned-squad --data_fn ../processed_data/model/propose/MED1250_filtered.tsv --out_fn ../processed_data/evaluate/MED1250/MED1250_bert-ab3p-ft_filtered
python model/predict.py --model ../processed_data/model/finetune_on_ab3p/checkpoint-final/ --tokenizer bert-large-cased-whole-word-masking-finetuned-squad --data_fn ../processed_data/model/propose/MED1250_proposal.tsv --out_fn ../processed_data/evaluate/MED1250/MED1250_bert-ab3p-ft_proposal

# Run SQuAD-fine-tuned BERT model on MED1250 data using proposed SF: 
python model/predict.py --model bert-large-cased-whole-word-masking-finetuned-squad --tokenizer bert-large-cased-whole-word-masking-finetuned-squad --data_fn ../processed_data/model/propose/MED1250_filtered.tsv --out_fn ../processed_data/evaluate/MED1250/MED1250_bert-squad-ft_filtered
python model/predict.py --model bert-large-cased-whole-word-masking-finetuned-squad --tokenizer bert-large-cased-whole-word-masking-finetuned-squad --data_fn ../processed_data/model/propose/MED1250_proposal.tsv --out_fn ../processed_data/evaluate/MED1250/MED1250_bert-squad-ft_proposal


# Run SQuAD-fine-tuned BERT model on MED1250 data using proposed SF,
# but only keep non-redundant LF-SF pairs.
# Also using all proposed (without fltr_answerable).
python model/predict.py --model bert-large-cased-whole-word-masking-finetuned-squad --tokenizer bert-large-cased-whole-word-masking-finetuned-squad --nonredundant --topk 20 --data_fn ../processed_data/model/propose/MED1250_all_proposal.tsv --out_fn ../processed_data/evaluate/MED1250/MED1250_bert-squad-ft_all_proposal_nonredundant


# Ken performed reranking: 
# Results are in /mnt/big/kwc/pubmed/Boxiang/forR/med1250/part3/MED1250_bert-squad-ft_all_proposal_nonredundant.best
cat /mnt/big/kwc/pubmed/Boxiang/forR/med1250/part3/MED1250_bert-squad-ft_all_proposal_nonredundant.best | python model/filter.py --ftype table --column 3 > test


# Compare various models on the MED1250 dataset:
python evaluate/med1250.py


# Rerank: 
# Ken performed reranking: /mnt/big/kwc/pubmed/Boxiang/rerank

# Remove punctuations from the left-edge of the reranked results: 
for f in `ls /mnt/big/kwc/pubmed/Boxiang/rerank/*rerank`; do
    base=`basename $f .rerank`
    cat $f | python3 model/rm_punc.py > ../processed_data/model/rm_punc/${base}_rm-punc
done


# Rejection modeling: 
cut -f1 ../processed_data/model/propose/MED1250_proposal.tsv | /mnt/big/kwc/pubmed/Boxiang/score_proposals.sh | sed "s/score/reject_score/" > ../processed_data/model/propose/MED1250_proposal_reject-score


# Get frequency:
cut -f1 ../processed_data/model/propose/MED1250_proposal.tsv | python model/freq.py > ../processed_data/model/propose/MED1250_proposal_freq



############
# Analysis #
############
# TBD:
python model/analysis/ab3p5bert.py 


