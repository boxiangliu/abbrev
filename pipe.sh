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


# Run Ab3P on BioC formatted data:
for f in ../data/BioC/*/*_bioc_gold.txt; do
    base=`basename $f`
    bash preprocess/ab3p/run_ab3p_simple.sh $f | sed "s/^  /ab3p:\t/" > ../processed_data/preprocess/ab3p/BioC/$base
done

# Reformat MED1250 to FASTA format:
python3 preprocess/med1250/text2fasta.py --med1250_fn "../data/MED1250/MED1250_labeled" --out_fn "../processed_data/preprocess/med1250/text2fasta/MED1250_labeled"
# The output file "../processed_data/preprocess/med1250/text2fasta/MED1250_labeled" has only 810 SF-LF pairs, but the original file "../data/MED1250/MED1250_labeled"
# has 1221 SF-LF pairs. What happened in the text2fasta.py script?

# Extract answerable sentences. 
# An answerable sentence is one with a short form and long form pair.
cat ../processed_data/preprocess/med1250/text2fasta/MED1250_labeled | python3 preprocess/med1250/fltr_answerable.py > ../processed_data/preprocess/med1250/fltr_answerable/MED1250_labeled


# Get the unlabeled dataset:
grep -v "^  " ../processed_data/preprocess/med1250/fltr_answerable/MED1250_labeled > ../processed_data/preprocess/med1250/fltr_answerable/MED1250_unlabeled


# Convert labeled dataset to tsv format:
python3 preprocess/med1250/fasta2table.py --in_fn ../processed_data/preprocess/med1250/fltr_answerable/MED1250_labeled --out_fn ../processed_data/preprocess/med1250/fltr_answerable/MED1250_labeled.tsv


# Get training PMIDs:
cut -f5 ../processed_data/preprocess/propose/pubmed19n0*.tsv | uniq | python3 preprocess/trng_PMIDs.py --med1250_pmid ../data/MED1250/MED1250_PMID --biotext_pmid ../data/BioText/BioText_PMID > ../processed_data/preprocess/trng_PMIDs/trng_PMID


# Converting Bio-C formatted XML to txt format:
# Ken did this. 
python3 preprocess/bioc/xml_to_txt.sh

# Create training examples for character RNN:
cat ../processed_data/preprocess/ab3p/summarize_ab3p/ab3p_freq.csv | python3 preprocess/character_rnn/data/ab3p2pos.py --exclude ../processed_data/preprocess/character_rnn/data/eval/Ab3P_pos,../processed_data/preprocess/character_rnn/data/eval/medstract_pos,../processed_data/preprocess/character_rnn/data/eval/bioadi_pos,../processed_data/preprocess/character_rnn/data/eval/SH_pos | sort -t$'\t' -k1,1 -k2,2 | uniq > ../processed_data/preprocess/character_rnn/data/train/pos
cat ../processed_data/preprocess/ab3p/summarize_ab3p/ab3p_freq.csv | python3 preprocess/character_rnn/data/ab3p2neg.py | sort -t$'\t' -k1,1 -k2,2 | uniq > ../processed_data/preprocess/character_rnn/data/train/neg
python3 preprocess/character_rnn/data/concat.py --pos ../processed_data/preprocess/character_rnn/data/train/pos --neg ../processed_data/preprocess/character_rnn/data/train/neg > ../processed_data/preprocess/character_rnn/data/train/concat

# Create evaluation examples for character RNN:
bash preprocess/bioc/bioc2eval.sh ../data/BioC/ ../processed_data/preprocess/character_rnn/data/eval/


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


# Train character RNN:
python3 model/character_rnn/train.py


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
cat /mnt/big/kwc/pubmed/Boxiang/forR/med1250/part3/MED1250_bert-squad-ft_all_proposal_nonredundant.best | python model/filter.py --ftype table --column 3 > ../processed_data/evaluate/MED1250/MED1250_bert-squad-ft_all_proposal_nonredundant.best.filtered


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
# Test RNN #
############
# Create toy example:
python3 model/character_rnn/example/toy_data.py > ../processed_data/model/character_rnn/example/toy_data/toy_data.tsv


########################
# Classify short forms #
########################
# Classify short forms into valid or invalid short forms.
for fn in `ls ../data/BioC/*/*bioc_gold.txt`; do
    base=`basename $fn _bioc_gold.txt`
    cat $fn | python3 preprocess/bioc/propose_sf_on_bioc.py > ../processed_data/preprocess/bioc/propose_sf_on_bioc/$base
done

# Train LSTM model: 
python3 model/character_rnn/train.py --config_fn ../processed_data/model/character_rnn/lstm/run_01/config.json
for i in {2..5}; do
    python3 model/character_rnn/train.py --config_fn ../processed_data/model/character_rnn/lstm/run_0${i}/config.json
done

# Output prediction result: 
python3 model/character_rnn/infer.py --model_fn ../processed_data/model/character_rnn/lstm/run_01/model.pt --eval_fn ../processed_data/preprocess/bioc/propose_sf_on_bioc/medstract --arch lstm > ../processed_data/model/character_rnn/lstm/run_01/preds.tsv
python3 model/character_rnn/infer.py --model_fn ../processed_data/model/character_rnn/lstm/run_02/model.pt --eval_fn ../processed_data/preprocess/bioc/propose_sf_on_bioc/medstract --arch lstm_embed > ../processed_data/model/character_rnn/lstm/run_02/preds.tsv
python3 model/character_rnn/infer.py --model_fn ../processed_data/model/character_rnn/lstm/run_03/model.pt --eval_fn ../processed_data/preprocess/bioc/propose_sf_on_bioc/SH --arch lstm_embed > ../processed_data/model/character_rnn/lstm/run_03/preds.tsv
python3 model/character_rnn/infer.py --model_fn ../processed_data/model/character_rnn/lstm/run_04/model.pt --eval_fn ../processed_data/preprocess/bioc/propose_sf_on_bioc/bioadi --arch lstm_embed > ../processed_data/model/character_rnn/lstm/run_04/preds.tsv
python3 model/character_rnn/infer.py --model_fn ../processed_data/model/character_rnn/lstm/run_05/model.pt --eval_fn ../processed_data/preprocess/bioc/propose_sf_on_bioc/Ab3P --arch lstm_embed > ../processed_data/model/character_rnn/lstm/run_05/preds.tsv


# Run proposal program version 2:
# This version extracts the following PSF-PLF pairs
# (PLF, PSF)
# PLF (PSF)
# PSF (PLF)
for fn in `ls ../data/BioC/*/*bioc_gold.txt`; do
    base=`basename $fn _bioc_gold.txt`
    cat $fn | python3 preprocess/bioc/propose_sf_on_bioc_2.py > ../processed_data/preprocess/bioc/propose_sf_on_bioc_2/$base
done

cat ../data/BioC/SH-BioC/SH_bioc_gold.txt | python3 preprocess/bioc/propose_sf_on_bioc_2.py > /dev/null



# Compare the potential LFs and SFs with the gold LFs and SFs:
for fn in `ls ../processed_data/preprocess/bioc/propose_sf_on_bioc_2/*`; do
    base=`basename $fn`
    cat $fn | python analysis/compare_potential_vs_gold.py > ../processed_data/analysis/compare_potential_vs_gold/$base
done


# Count the number of each type of missing pair:
grep -P "\tSF\t" ../processed_data/analysis/compare_potential_vs_gold/* | wc
grep -P "\tSF\tmissing SF\t" ../processed_data/analysis/compare_potential_vs_gold/* | wc
grep -P "\tSF\tgap\t" ../processed_data/analysis/compare_potential_vs_gold/* | wc
grep -P "\tSF\tmulti-span LF\t" ../processed_data/analysis/compare_potential_vs_gold/* | wc
grep -P "\tSF\tother\t" ../processed_data/analysis/compare_potential_vs_gold/* | wc
# total: 123
# missing SF: 56
# gap: 24
# multi-span LF: 38
# other: 5


# run proposal program version 2 on ab3p results:
for fn in `ls ../processed_data/preprocess/ab3p/BioC/*_bioc_gold.txt`; do
    base=`basename $fn _bioc_gold.txt`
    cat $fn | python3 preprocess/bioc/propose_sf_on_bioc_2.py > ../processed_data/preprocess/bioc/propose_sf_on_bioc_2/Ab3P_processed/$base
done


# Compare the potential LFs and SFs with the gold LFs and SFs:
for fn in `ls ../processed_data/preprocess/bioc/propose_sf_on_bioc_2/Ab3P_processed/*`; do
    base=`basename $fn`
    cat $fn | python analysis/compare_potential_vs_Ab3P.py > ../processed_data/analysis/compare_potential_vs_Ab3P/$base
done


# Compare the Ab3P SFs and LFs with the gold LFs and SFs:
for fn in `ls ../processed_data/preprocess/bioc/propose_sf_on_bioc_2/Ab3P_processed/*`; do
    base=`basename $fn`
    cat $fn | python analysis/compare_Ab3P_vs_gold.py > ../processed_data/analysis/compare_Ab3P_vs_gold/$base
done


# Count the number of each type of missing pair:
grep -P "\tSF\t" ../processed_data/analysis/compare_Ab3P_vs_gold/* | wc
grep -P "\tSF\tmissing SF\t" ../processed_data/analysis/compare_Ab3P_vs_gold/* | wc
grep -P "\tSF\tgap\t" ../processed_data/analysis/compare_Ab3P_vs_gold/* | wc
grep -P "\tSF\tmulti-span LF\t" ../processed_data/analysis/compare_Ab3P_vs_gold/* | wc
grep -P "\tSF\tother\t" ../processed_data/analysis/compare_Ab3P_vs_gold/* | wc
# total: 751
# missing SF: 701
# gap: 2
# multi-span LF: 1
# other: 47

############
# NER data #
############
# generate BIO data:
for fn in `ls ../processed_data/preprocess/bioc/propose_sf_on_bioc_2/{Ab3P,bioadi,medstract,SH}`; do
    base=`basename $fn`
    cat $fn | python3 preprocess/NER/BIO_data.py > ../processed_data/preprocess/NER/BIO_data/$base
done


####################
# QA and rejection #
####################
# The QA and rejection experiment does the following
# 1. propose short forms and questions
# 2. use BERT QA to propose long forms 
# 3. use character-based LSTM to reject. 

# Propose short form, questions, and answers.
for fn in `ls ../data/BioC/*/*bioc_gold.txt`; do
    base=`basename $fn _bioc_gold.txt`
    cat $fn | python3 preprocess/bioc/propose_qa_on_bioc.py > ../processed_data/preprocess/bioc/propose_qa_on_bioc/$base
done


# Predict on proposed short forms:
# Note that Ab3P will throw an "Insufficient memory" error. Use CPU instead of GPU for Ab3P.
for fn in `ls ../processed_data/preprocess/bioc/propose_qa_on_bioc/{Ab3P,bioadi,medstract,SH}`; do
    base=`basename $fn`; echo $base
    sbatch -p  1080Ti,1080Ti_mlong,1080Ti_slong,2080Ti,2080Ti_mlong,M40x8,M40x8_mlong,M40x8_slong,P100,TitanXx8,TitanXx8_mlong,TitanXx8_slong,V100_DGX,V100x8 \
        --gres=gpu:1 --job-name=$base --output=../processed_data/preprocess/bioc/propose_qa_on_bioc/${base}.log \
        --wrap "python model/predict.py --model bert-large-cased-whole-word-masking-finetuned-squad --tokenizer bert-large-cased-whole-word-masking-finetuned-squad \
        --nonredundant --topk 10 --data_fn $fn --out_fn ../processed_data/preprocess/bioc/propose_qa_on_bioc/${base}.fasta"
done

for fn in `ls ../processed_data/preprocess/bioc/propose_qa_on_bioc/{Ab3P,bioadi,medstract,SH}.fasta`; do
    base=`basename $fn .fasta`; echo $base
    cat $fn | python3 model/qa_reject/QA_output_to_LSTM_input.py > ../processed_data/model/qa_reject/QA_output_to_LSTM_input/${base}
done


# Train SF-LF model with cross validation, using one of four datasets as the testset.
python3 model/qa_reject/train_2_losses.py --config_fn ../processed_data/model/qa_reject/lstm/run_01/config.json
python3 model/qa_reject/train_2_losses.py --config_fn ../processed_data/model/qa_reject/lstm/run_02/config.json
python3 model/qa_reject/train_2_losses.py --config_fn ../processed_data/model/qa_reject/lstm/run_03/config.json
python3 model/qa_reject/train_2_losses.py --config_fn ../processed_data/model/qa_reject/lstm/run_04/config.json

# Infer: 
python3 model/qa_reject/infer_2_losses.py --model_fn ../processed_data/model/qa_reject/lstm/run_01/model.pt --eval_fn ../processed_data/model/qa_reject/QA_output_to_LSTM_input/medstract --arch lstm_embed > ../processed_data/model/qa_reject/lstm/run_01/preds.tsv


# Generate toy SF-LF data:
# SF will be the first letter from each word in the long form.
python3 model/qa_reject/toy_data.py --n_examples 10000 > ../processed_data/model/qa_reject/toy_data/train
python3 model/qa_reject/toy_data.py --n_examples 1000 > ../processed_data/model/qa_reject/toy_data/test

# Generate toy SF-LF data:
# SF will be a random letter from each word in the long form.
python3 model/qa_reject/toy_data2.py --n_examples 10000 > ../processed_data/model/qa_reject/toy_data2/train
python3 model/qa_reject/toy_data2.py --n_examples 1000 > ../processed_data/model/qa_reject/toy_data2/test


# Generate toy SF-LF data:
# SF will be a random letter from each word in the long form.
# The length of LF varies from 2 to 10 words.
# The length of each word varies from 2 to 10 characters.
# The SF has a 0.9 probability of getting a letter from each word of the LF. 
python3 model/qa_reject/toy_data3.py --n_examples 10000 > ../processed_data/model/qa_reject/toy_data3/train
python3 model/qa_reject/toy_data3.py --n_examples 1000 > ../processed_data/model/qa_reject/toy_data3/test


# Train on toy data:
# toy_01: Use the ToyEmbedRNN architecture. NO attention.
python3 model/qa_reject/toy_train.py --config_fn ../processed_data/model/qa_reject/lstm/toy_01/config.json

# toy_02: Use the ToyEmbedRNNSequence architecture. With attention.
python3 model/qa_reject/toy_train.py --config_fn ../processed_data/model/qa_reject/lstm/toy_02/config.json

# toy_03: Use the new toy data. Each character of the SF is a random letter from the LF.
python3 model/qa_reject/toy_train.py --config_fn ../processed_data/model/qa_reject/lstm/toy_03/config.json

# toy_04: Use the ToyEmbedRNNSequenceAvg architecture. Average over the entire sequence.
python3 model/qa_reject/toy_train.py --config_fn ../processed_data/model/qa_reject/lstm/toy_04/config.json

# toy_05: Use the ToyEmbedRNNSequenceAvg architecture. Also use toy_data3: LF length in (2, 10), word length in (2, 10), LF has 0.9 probability to inherit a letter from each word of LF.
python3 model/qa_reject/toy_train.py --config_fn ../processed_data/model/qa_reject/lstm/toy_05/config.json

# toy_06: larger learning rate. large batch size, 5x more epochs:
python3 model/qa_reject/toy_train.py --config_fn ../processed_data/model/qa_reject/lstm/toy_06/config.json


# Train a new SF-LF model:
python3 model/qa_reject/train_1_loss.py --config_fn ../processed_data/model/qa_reject/lstm/run_05/config.json
python3 model/qa_reject/infer_1_loss.py --model_fn ../processed_data/model/qa_reject/lstm/run_05/checkpoint-050000.pt --eval_fn ../processed_data/model/qa_reject/QA_output_to_LSTM_input/Ab3P --arch EmbedRNNSequenceAvg > ../processed_data/model/qa_reject/lstm/run_05/preds.tsv

########################
# Ab3P on SQuAD output #
########################
for fn in `ls ../processed_data/preprocess/bioc/propose_qa_on_bioc/*.fasta`; do
    base=`basename $fn .fasta`
    cat $fn | python3 model/qa_to_ab3p/qa_to_ab3p-input.py > ../processed_data/model/qa_to_ab3p/qa_to_ab3p-input/$base 
    bash preprocess/ab3p/run_ab3p_simple.sh ../processed_data/model/qa_to_ab3p/qa_to_ab3p-input/$base | python3 model/qa_to_ab3p/parse_ab3p_output.py > ../processed_data/model/qa_to_ab3p/parse_ab3p_output/$base
done


######################
# Suffix frequencies #
######################
# Generate suffix frequencies.
# Copied from Ken's directory, 
bash model/suffix_freq/get_suffix_freqs.sh 

# Parse suffix frequencies:
cat ../processed_data/model/suffix_freq/get_suffix_freqs/suffix_freqs.txt | python3 model/suffix_freq/parse_suffix_freqs.py > ../processed_data/model/suffix_freq/parse_suffix_freqs/suffix_freqs.tsv


#######################
# logistic regression #
#######################
# Merge frequencies and Ab3P result:
python3 model/logistic_regression/merge_data_sources.py



#################
# Seq2seq model #
#################
# Create NER dataset:
for fn in `ls ../data/BioC/*/*bioc_gold.txt`; do
    base=`basename $fn _bioc_gold.txt`
    cat $fn | python3 preprocess/bioc/bioc2ner.py > ../processed_data/preprocess/bioc/bioc2ner/$base
done


############
# Analysis #
############
# TBD:
cat ../data/BioC/Ab3P-BioC/Ab3P_bioc_gold.txt | python model/analysis/ab3p5bert.py


# Compare Ab3P results with the gold standard:
for fn in `ls ../processed_data/preprocess/ab3p/BioC/*_bioc_gold.txt`; do
    base=`basename $fn _bioc_gold.txt`
    cat $fn | python3 analysis/ab3p/ab3p_vs_gold.py > ../processed_data/analysis/ab3p/ab3p_vs_gold/$base
done


# Compare BERT SQuAD results with the gold standard:
for fn in `ls ../processed_data/model/qa_reject/QA_output_to_LSTM_input/{Ab3P,bioadi,SH,medstract}`; do
    base=`basename $fn`
    python analysis/squad/squad_vs_gold.py --proposal ../processed_data/preprocess/bioc/propose_qa_on_bioc/$base --squad $fn > ../processed_data/analysis/squad/squad_vs_gold/$base
done


# Examine cases missed by the short form proposal program: 
awk -F $'\t' '$3 == 1 && $4 == "omit"' ../processed_data/preprocess/bioc/propose_qa_on_bioc/{Ab3P,bioadi,SH,medstract} | less 


# How many words are within short forms?
grep -P "\tSF" ../data/BioC/*/*_bioc_gold.txt | cut -f3,3 | awk '{print length, NF, $0}' | sort -k2,2n 
# Answer: 
# Total number of short forms: 4081
# 1 word: 3972 (0.9732)
# 2 words: 96 (0.0235)
# 3 words: 11 (0.0026)
# 4 words: 1 (0.0002)
# >5 words: 0


grep -P "\tLF" ../data/BioC/*/*_bioc_gold.txt | cut -f3,3 | awk '{print length, NF, $0}' | sort -k2,2n 