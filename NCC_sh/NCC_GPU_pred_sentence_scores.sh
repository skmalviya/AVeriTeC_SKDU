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
test_ckpt=RobertaCls_0519_11:32:53_sent_reranker_multi_10

cd /media/STORAGE/Shrikant/AVeriTeC_SKDU

PYTHONPATH=src /home/shrikant/miniconda3/envs/averitec_DESK1/bin/python \
src/my_methods/roberta_sentence_selector/pred_sentence_scores.py \
--data_dir data \
--batch_size 64 \
--bert_name roberta-base \
--test_ckpt $test_ckpt
 2>&1 | tee NCC_err_out_logs/NCC_GPU_pred_sentence_scores.${test_ckpt}.log




echo "End Time: $(date)"

