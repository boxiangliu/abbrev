for input in Ab3P_bioc bioadi_bioc medstract_bioc SH_bioc; do
    input_=/mnt/scratch/boxiang/projects/abbrev/processed_data/frontiers/${input}.txt
    echo $input_
    for model in bert bart biobert t5 ernie; do
        model_=/mnt/scratch/boxiang/projects/bert_acronym.py/${model}_acronym.py
        echo $model_
        cat $input_ | python $model_ > /mnt/scratch/boxiang/projects/abbrev/processed_data/frontiers/${input}_${model}.txt
    done
done