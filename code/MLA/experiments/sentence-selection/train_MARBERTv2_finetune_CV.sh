#!/bin/bash
#
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Authors: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.
#
#SBATCH --job-name=train_bert-base
#SBATCH --out='train_bert-base.log'
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:tesla_a100:1

conda_setup="/home/smg/$(whoami)/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${conda_setup}" ]]; then
  #shellcheck disable=SC1090
  . "${conda_setup}"
  conda activate mla
fi

set -ex

task='sentence-selection'
pretrained='UBC-NLP/MARBERTv2'
max_len=128
# lr=2e-5
# epochs_num=2
# fold=1
for fold in 1 2 3 4 5
do 
 for epochs_num in 2 3 4 5
  do
   for lr in 2e-5 3e-5 4e-5 5e-5
    do 
     model_dir="CV_experiments/MARBERTv2-${max_len}-mod_4neg_lr${lr}_${epochs_num}epochs_fold${fold}"
     out_dir="CV_experiments/MARBERTv2-${max_len}-out_4neg_lr${lr}_${epochs_num}epochs_fold${fold}"
     inp_dir="CV_experiments/MARBERTv2-${max_len}-inp_4neg_lr${lr}_${epochs_num}epochs_fold${fold}"

     data_dir='/data/'
     model='base'

     if [[ -d "${model_dir}" ]]; then
      echo "${model_dir} exists!"
      exit
     fi

     mkdir -p "${inp_dir}"

     python '../../preprocess_sentence_selection_AuRED_CV.py' \
      --in_file "${data_dir}/AuRED.json" \
      --neg_ratio 4 \
      --fold_file "${data_dir}/training_folds/train_fold${fold}.tsv" \
      --out_file "${inp_dir}/train.tsv" \
      --training


     python '../../train.py' \
      --task "${task}" \
      --data_dir "${inp_dir}" \
      --default_root_dir "${model_dir}" \
      --pretrained_model_name "${pretrained}" \
      --max_seq_length "${max_len}" \
      --model_name "${model}" \
      --use_title \
      --max_epochs "${epochs_num}" \
      --train_batch_size 8\
      --eval_batch_size 8 \
      --accumulate_grad_batches 8 \
      --learning_rate "${lr}" \
      --warmup_ratio 0.06 \
      --adafactor \
      --gradient_clip_val 1.0 \
      --precision 16 \
      --deterministic true \
      --gpus 1
    done
  done
done    
