#!/bin/bash
#
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Authors: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.
#
#SBATCH --job-name=predict_bert-base
#SBATCH --out='predict_bert-base.log'
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:tesla_a100:1

conda_setup="/home/smg/$(whoami)/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${conda_setup}" ]]; then
  #shellcheck disable=SC1090
  . "${conda_setup}"
  conda activate mla
fi

set -ex

pretrained='UBC-NLP/MARBERTv2'
max_len=128
fold=5
#learning rate for best dev on the sentence selection task (just to open the pred_sent files )
pred_lr=4e-5 #[4e-5,5e-5,4e-5,5e-5,4e-5]
#Best learning rate for the claim verification model
lr=5e-5 #[2e-5,4e-5,4e-5,5e-5,5e-5]

model_dir="MARBERTv2-${max_len}-mod_4neg_lr${lr}_5epochs_fold${fold}"
out_dir="MARBERTv2-GoldSet"
data_dir="/data/"
pred_sent_dir="../sentence-selection/CV_experiments/MARBERTv2-128-out_4neg_lr${pred_lr}_5epochs_fold${fold}"

unset -v latest

for file in "${model_dir}/checkpoints"/*.ckpt; do
  [[ $file -nt $latest ]] && latest=$file
done

if [[ -z "${latest}" ]]; then
  echo "Cannot find any checkpoint in ${model_dir}"
  exit
fi

echo "Latest checkpoint is ${latest}"

mkdir -p "${out_dir}"

split='test'

if [[ -f "${out_dir}/${split}.jsonl" ]]; then
  echo "Result '${out_dir}/${split}.jsonl' exists!"
  exit
fi

python '../../preprocess_claim_verification_AuRED.py' \
  --corpus "${data_dir}/AuRED_star.json" \
  --in_file "${pred_sent_dir}/MLA_test_4neg_lr${pred_lr}_5epochs_fold${fold}_GoldSet.jsonl" \
  --out_file "${out_dir}/${split}_gold_${fold}.tsv"

 python '../../predict.py' \
  --checkpoint_file "${latest}" \
  --in_file "${out_dir}/${split}_gold_${fold}.tsv" \
  --out_file "${out_dir}/${split}_gold_${fold}.out" \
  --batch_size 128 \
  --gpus 1

 python '../../postprocess_claim_verification_AuRED.py' \
  --data_file "${data_dir}/AuRED_star.json" \
  --fold_file "${data_dir}/training_folds/test_fold${fold}.tsv" \
  --pred_sent_file "${pred_sent_dir}/MLA_test_4neg_lr${pred_lr}_5epochs_fold${fold}_GoldSet.jsonl" \
  --pred_claim_file "${out_dir}/${split}_gold_${fold}.out" \
  --out_file "${out_dir}/${split}_gold_fold${fold}.jsonl"

