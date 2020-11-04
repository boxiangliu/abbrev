data_fn=$1
out_dir=$2
checkpoint_fn=$3
n_line_per_chunk=${4:-500}

echo -e "Data\t$data_fn"
echo -e "Outdir\t$out_dir"
echo -e "Checkpoint\t$checkpoint_fn"
echo -e "$n_line_per_chunk lines per chunk"

mkdir -p $out_dir

header_fn=$out_dir/header
body_fn=$out_dir/body

head -n 1 $data_fn > $header_fn
tail -n +2 $data_fn > $body_fn

split -l $n_line_per_chunk $body_fn $out_dir/_

for fn in `ls $out_dir/_*`; do
    base=`basename $fn`
    cat $header_fn $fn > $out_dir/chunk${base}.in
    rm $fn
done
rm $header_fn
rm $body_fn

for fn in `ls $out_dir/chunk_*.in`; do
    base=`basename $fn .in`
    sbatch -p 1080Ti,1080Ti_mlong,1080Ti_short,1080Ti_slong,2080Ti,2080Ti_mlong,M40x8,M40x8_mlong,M40x8_slong,P100,TitanXx8,TitanXx8_short,TitanXx8_mlong,TitanXx8_slong,V100_DGX,V100x8 \
        --gres=gpu:1 --job-name=$base --output=$out_dir/${base}.log \
        --wrap "python model/predict.py \
        --model $checkpoint_fn \
        --tokenizer bert-large-cased-whole-word-masking-finetuned-squad \
        --data_fn $fn --out_fn $out_dir/${base}.out"
done