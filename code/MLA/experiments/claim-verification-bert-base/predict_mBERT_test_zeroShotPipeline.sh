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


model_dir="PATH for model fine tuned with FEVER/bert-base-multilingual-uncased-128-mod"
out_dir="bert-base-multilingual-uncased-128-mod_zeroShotPipeline"

data_dir="/data/"
pred_sent_dir="/data/experiments/sentence-selection/multilingual_experiments/MLA-FEVER_zeroShot/bert-base-multilingual-uncased-128-out"

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
  --corpus "${data_dir}/AuRED.json" \
  --in_file "${pred_sent_dir}/MLA_FEVER_zeroShot_fold${fold}.jsonl" \
  --out_file "${out_dir}/${split}.tsv"

 python '../../predict.py' \
  --checkpoint_file "${latest}" \
  --in_file "${out_dir}/${split}.tsv" \
  --out_file "${out_dir}/${split}.out" \
  --batch_size 128 \
  --gpus 1

 python '../../postprocess_claim_verification_AuRED.py' \
  --data_file "${data_dir}/AuRED_data.json" \
  --fold_file "${data_dir}/training_folds/test_fold${fold}.tsv" \
  --pred_sent_file "${pred_sent_dir}/MLA_FEVER_zeroShot_fold${fold}.jsonl" \
  --pred_claim_file "${out_dir}/${split}.out" \
  --out_file "${out_dir}/${split}_fold${fold}.jsonl"

#  python '../../eval_fever.py' \
#   --gold_file "${data_dir}/AuRED_data.json" \
#   --fold_file "${data_dir}/training_folds/dev_fold${fold}.tsv" \
#   --pred_file "${out_dir}/${split}.jsonl" \
#   --out_file "${out_dir}/eval.${split}.txt"
