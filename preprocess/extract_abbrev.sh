out_dir="../processed_data/preprocess/abbrev/"
mkdir -p $out_dir

extract_abbrev(){
    in_fn=$1
    out_dir=$2

    echo -e "INPUT\t$in_fn"
    out_fn="$out_dir/`basename $in_fn`"
    echo -e "OUTPUT\t$out_fn"
    python3 preprocess/extract_abbrev.py --in_fn $in_fn --out_fn $out_fn

}
export -f extract_abbrev

parallel -j 35 --joblog ../processed_data/preprocess/abstract/extract_abrev.log extract_abbrev {} {} ::: `ls ../processed_data/preprocess/sentence/pubmed*.txt` ::: $out_dir
