#!/bin/bash
#
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Authors: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.
#
#SBATCH --job-name=train_bert-base
#SBATCH --out='train_bert-base.log'
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:tesla_a100:1

conda_setup="/home/smg/$(whoami)/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${conda_setup}" ]]; then
  #shellcheck disable=SC1090
  . "${conda_setup}"
  conda activate mla
fi

set -ex

task='claim-verification'
pretrained='UBC-NLP/MARBERTv2'
max_len=128
fold=5
#learning rate for best dev on the sentence selection task (just to open the pred_sent files )
#the best lr for evidence reteival on the specific fold 
pred_lr=4e-5
#learning rate to train the claim verification model
for lr in 2e-5 3e-5 4e-5 5e-5
do
 model_dir="MARBERTv2-${max_len}-mod_4neg_lr${lr}_5epochs_fold${fold}"
 inp_dir="MARBERTv2-${max_len}-inp_4neg_lr${lr}_5epochs_fold${fold}"


 data_dir="/data/"
 pred_sent_dir="../sentence-selection/CV_experiments/MARBERTv2-128-out_4neg_lr${pred_lr}_5epochs_fold${fold}"

 model='verification-joint'
 aggregate_mode='attn'
 attn_bias_type='value_only'

 if [[ -d "${model_dir}" ]]; then
  echo "${model_dir} exists! Skip training."
  exit
 fi

 mkdir -p "${inp_dir}"

 python '../../preprocess_claim_verification_AuRED.py' \
  --corpus "${data_dir}/AuRED.json" \
  --in_file "${pred_sent_dir}/MLA_train_4neg_lr${pred_lr}_5epochs_fold${fold}.jsonl" \
  --out_file "${inp_dir}/train.tsv" \
  --training

 python '../../train.py' \
  --task "${task}" \
  --data_dir "${inp_dir}" \
  --default_root_dir "${model_dir}" \
  --pretrained_model_name "${pretrained}" \
  --max_seq_length "${max_len}" \
  --model_name "${model}" \
  --aggregate_mode "${aggregate_mode}" \
  --attn_bias_type "${attn_bias_type}" \
  --sent_attn \
  --word_attn \
  --class_weighting \
  --use_title \
  --max_epochs 5 \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --accumulate_grad_batches 8 \
  --learning_rate ${lr} \
  --warmup_ratio 0.06 \
  --adafactor \
  --gradient_clip_val 1.0 \
  --precision 16 \
  --deterministic true \
  --gpus 1
done