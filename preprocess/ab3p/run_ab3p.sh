wd=/mnt/scratch/boxiang/projects/Ab3P/
cd $wd

in_dir=/mnt/scratch/boxiang/projects/abbrev/processed_data/preprocess/sentence/
out_dir=/mnt/scratch/boxiang/projects/abbrev/processed_data/preprocess/ab3p/run_ab3p/
mkdir -p $out_dir

ab3p(){
    in_fn=$1
    out_dir=$2
    echo -e "INPUT\t$in_fn"
    base=`basename $in_fn .txt`
    tmp_fn=$out_dir/$base.in
    out_fn=$out_dir/$base.out

    awk 'BEGIN {FS = "|"}; {print ">"$1"|"$2"|"$3"\n"$4}' $in_fn > $tmp_fn
    ./identify_abbr $tmp_fn &> $out_fn
}
export -f ab3p

parallel -j 35 ab3p {} {} ::: `ls $in_dir/pubmed*.txt` ::: $out_dir