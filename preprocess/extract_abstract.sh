out_dir=../processed_data/preprocess/abstract/
mkdir -p $out_dir

for in_fn in `ls /mnt/big/kwc/pubmed/raw/*.xml.gz`; do
    echo -e "INPUT\t$in_fn"
    out_fn="$out_dir/`basename $in_fn .xml.gz`.txt"
    echo -e "OUTPUT\t$out_fn"
    python3 preprocess/extract_abstract.py --in_fn $in_fn --out_fn $out_fn
done