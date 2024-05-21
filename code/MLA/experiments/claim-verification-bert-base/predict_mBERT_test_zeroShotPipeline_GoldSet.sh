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

pretrained='bert-base-multilingual-uncased'
max_len=128
fold=5
#learning rate for best dev on the sentence selection task (just to open the pred_sent files )[4e-5,5e-5,4e-5,5e-5,4e-5]
# pred_lr=4e-5


model_dir="PATH for model fine tuned with FEVER/bert-base-multilingual-uncased-128-mod"
out_dir="bert-base-multilingual-uncased-128-mod_zeroShotPipeline"

data_dir="/data/"
pred_sent_dir="/experiments/sentence-selection/multilingual_experiments/MLA-FEVER_zeroShot/bert-base-multilingual-uncased-128-out"

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
  --in_file "${pred_sent_dir}/MLA_FEVER_zeroShot_fold${fold}_GoldSet.jsonl" \
  --out_file "${out_dir}/${split}_gold.tsv"

 python '../../predict.py' \
  --checkpoint_file "${latest}" \
  --in_file "${out_dir}/${split}_gold.tsv" \
  --out_file "${out_dir}/${split}_gold.out" \
  --batch_size 128 \
  --gpus 1

 python '../../postprocess_claim_verification_AuRED.py' \
  --data_file "${data_dir}/AuRED_star.json" \
  --fold_file "${data_dir}/training_folds/test_fold${fold}.tsv" \
  --pred_sent_file "${pred_sent_dir}/MLA_FEVER_zeroShot_fold${fold}_GoldSet.jsonl" \
  --pred_claim_file "${out_dir}/${split}_gold.out" \
  --out_file "${out_dir}/${split}_gold_fold${fold}.jsonl"


