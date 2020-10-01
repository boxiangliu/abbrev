out_dir=../processed_data/preprocess/abstract/
mkdir -p $out_dir

extract_abstract(){
    in_fn=$1
    out_dir=$2

    echo -e "INPUT\t$in_fn"
    out_fn="$out_dir/`basename $in_fn .xml.gz`.txt"
    echo -e "OUTPUT\t$out_fn"
    python3 preprocess/extract_abstract.py --in_fn $in_fn --out_fn $out_fn

}
export -f extract_abstract

parallel -j 35 --joblog ../processed_data/preprocess/abstract/extract_abstract.log extract_abstract {} {} ::: `ls /mnt/big/kwc/pubmed/raw/*.xml.gz` ::: $out_dir
