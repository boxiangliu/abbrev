data_fn="../processed_data/preprocess/model/train_val/val.tsv"
out_dir="../processed_data/preprocess/model/predict/"
mkdir -p $out_dir

header_fn=$out_dir/header
body_fn=$out_dir/body

head -n 1 $data_fn > $header_fn
tail -n +2 $data_fn > $body_fn

split -l 100 $body_fn $out_dir/_

for fn in `ls $out_dir/_*`; do
    base=`basename $fn`
    cat $header_fn $fn > $out_dir/chunk${base}.in
    rm $fn
done
rm $header_fn
rm $body_fn

for fn in `ls $out_dir/chunk_*.in`; do
    base=`basename $fn .in`
    sbatch -p TitanXx8 --gres=gpu:1 --job-name=$base --output=$out_dir/${base}.log \
        --wrap "python model/predict.py \
        --model /mnt/scratch/boxiang/projects/abbrev/processed_data/model/finetune_on_ab3p/checkpoint-14500/ \
        --tokenizer bert-large-cased-whole-word-masking-finetuned-squad \
        --data_fn $fn --out_fn $out_dir/${base}.out"
done