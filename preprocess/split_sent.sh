in_dir=../processed_data/preprocess/abstract/
out_dir=../processed_data/preprocess/sentence/
mkdir -p $out_dir

split_sent(){
    in_fn=$1
    out_dir=$2
    out_fn="$out_dir/`basename $in_fn`"
    echo -e "INPUT\t$in_fn"
    echo -e "OUTPUT\t$out_fn"

    python3 preprocess/split_sent.py --in_fn $in_fn --out_fn $out_fn
}

export -f split_sent

parallel -j 10 --joblog $out_dir/split_sent.log split_sent {} {} ::: `ls $in_dir/pubmed*.txt` ::: $out_dir

