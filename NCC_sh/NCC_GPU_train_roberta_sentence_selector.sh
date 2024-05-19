#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:qmpere:1
#SBATCH -p res-gpu-small
#SBATCH --qos=long-high-prio    
#SBATCH -t 05-00:00:00
#SBATCH --job-name=Shri
#SBATCH -e NCC_err_out_logs/NCC_GPU_train_roberta_sentence_selector_stderr.txt
#SBATCH -o NCC_err_out_logs/NCC_GPU_train_roberta_sentence_selector_stdout.txt

#source /etc/profile
#module load cuda/8.0

#conda activate base

echo "Start Time: $(date)"

model_suffix=sent_reranker_multi_10
train_data_extend_multi=10

cd /media/STORAGE/Shrikant/AVeriTeC_SKDU

PYTHONPATH=src /home/shrikant/miniconda3/envs/averitec_DESK1/bin/python \
src/my_methods/roberta_sentence_selector/train_roberta_sentence_selector.py \
--data_dir data \
--bert_name roberta-base \
--max_epoch 10 \
--train_data_extend_multi $train_data_extend_multi \
--user_given_model_suffix $model_suffix \
 2>&1 | tee NCC_err_out_logs/NCC_GPU_train_roberta_sentence_selector.${model_suffix}.train_data.${train_data_extend_multi}.log




echo "End Time: $(date)"

