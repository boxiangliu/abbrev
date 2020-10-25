SQUAD_DIR=/mnt/scratch/boxiang/projects/abbrev/data/SQuAD2.0/
TRANSFORMERS_DIR=/mnt/scratch/boxiang/projects/transformers/

python -m torch.distributed.launch --nproc_per_node=10 $TRANSFORMERS_DIR/examples/question-answering/run_squad.py \
    --model_type bert \
    --model_name_or_path bert-large-cased-whole-word-masking \
    --do_train \
    --do_eval \
    --train_file $SQUAD_DIR/train-v2.0.json \
    --predict_file $SQUAD_DIR/dev-v2.0.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ../processed_data/experiments/run_squad/ \
    --per_gpu_eval_batch_size=3   \
    --per_gpu_train_batch_size=2   \
    --save_steps 5000 


SQUAD_DIR=/mnt/scratch/boxiang/projects/abbrev/data/SQuAD2.0/
TRANSFORMERS_DIR=/mnt/scratch/boxiang/projects/transformers/

python $TRANSFORMERS_DIR/examples/question-answering/run_squad.py \
    --model_type bert \
    --model_name_or_path bert-large-cased-whole-word-masking \
    --do_eval \
    --predict_file $SQUAD_DIR/dev-v2.0.json \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ../processed_data/experiments/run_squad/eval_squad/ \
    --per_gpu_eval_batch_size=128