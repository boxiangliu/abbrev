in_dir=../processed_data/preprocess/abstract/
out_dir=../processed_data/preprocess/sentence/
mkdir -p $out_dir

make_sub(){
    in_fn=$1
    out_dir=$2
    base=`basename $in_fn .txt`
    echo \#\!/bin/sh > $out_dir/${base}.sub
    echo "#SBATCH --gres=gpu:1" >> $out_dir/${base}.sub
    echo "#SBATCH --job-name=$base" >> $out_dir/${base}.sub 
    echo "#SBATCH --output=$out_dir/${base}.out" >> $out_dir/${base}.sub
    echo "python3 preprocess/split_sent.py --in_fn $in_fn --out_fn $out_dir/${base}.txt" >> $out_dir/${base}.sub
}
export -f make_sub

parallel -j 10 make_sub {} {} ::: `ls $in_dir/pubmed*.txt` ::: $out_dir


for sub in `ls $out_dir/pubmed*.sub`; do
    sbatch -p TitanXx8 $sub
done