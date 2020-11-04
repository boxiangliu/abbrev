in_dir=$1
out_dir=$2
mkdir -p $out_dir

echo -e "INPUT DIR\t$in_dir"
echo -e "OUTPUT DIR\t$out_dir"
echo 

for fn in `ls ${in_dir}/pubmed*.txt`; do
    echo -e "INPUT FILE\t$fn"
    base=`basename $fn .txt`

    sbatch -p CPUx40,1080Ti,1080Ti_mlong,1080Ti_short,1080Ti_slong,2080Ti,2080Ti_mlong,M40x8,M40x8_mlong,M40x8_slong,P100,TitanXx8,TitanXx8_short,TitanXx8_mlong,TitanXx8_slong,V100_DGX,V100x8 \
    --job-name=$base --output=$out_dir/${base}.log \
    --wrap "cat $fn | python preprocess/line2fasta.py | python model/propose.py > $out_dir/${base}.txt"
done