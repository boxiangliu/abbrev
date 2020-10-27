in_dir=${in_dir:-""}
out_dir=${out_dir:-""}
in_fn=${in_fn:-""}
out_fn=${out_fn:-""}

while [ $# -gt 0 ]; do

    if [[ $1 == *"--"* ]]; then
        v="${1/--/}"
        declare $v="$2"
    fi

    shift
done

wd=/mnt/scratch/boxiang/projects/Ab3P/
cd $wd


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

if [[ ! -z $in_dir ]] && [[ ! -z $out_dir ]]; then
    echo -e "INPUT DIR\t${in_dir}"
    echo -e "OUTPUT DIR\t${out_dir}"
    mkdir -p $out_dir

    parallel -j 35 ab3p {} {} ::: `ls $in_dir/pubmed*.txt` ::: $out_dir
fi


if [[ ! -z $in_fn ]] && [[ ! -z $out_fn ]]; then
    echo -e "INPUT FILE\t${in_fn}"
    echo -e "OUTPUT FILE\t${out_fn}"

    ./identify_abbr $in_fn &> $out_fn
fi
