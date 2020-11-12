in_dir=$1
out_dir=$2

# in_dir=../data/BioC/
# out_dir=../processed_data/preprocess/character_rnn/data/eval/

for f in `find $in_dir -name "*_bioc_gold.txt"`; do
    echo "FILE    $f"
    base=`basename $f _bioc_gold.txt`
    cat $f | python3 preprocess/bioc/bioc2pos.py > $out_dir/${base}_pos
done