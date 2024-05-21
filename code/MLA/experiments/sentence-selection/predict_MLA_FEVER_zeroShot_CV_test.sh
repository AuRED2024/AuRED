#!/bin/bash
#
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Authors: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.
#
#SBATCH --job-name=predict_bert-base_test
#SBATCH --out='predict_bert-base_test.log'
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
# lr=4e-5
# epochs_num=5
model_dir="multilingual_experiments/MLA-FEVER_zeroShot/bert-base-multilingual-uncased-128-mod"
out_dir="multilingual_experiments/MLA-FEVER_zeroShot/bert-base-multilingual-uncased-128-out"

data_dir='/data/'

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


if [[ -f "${out_dir}/${split}.jsonl" ]]; then
  echo "${out_dir}/${split}.jsonl exists!"
  exit
fi

python '../../preprocess_sentence_selection_AuRED_CV.py' \
  --in_file "${data_dir}/AuRED.json" \
  --fold_file "${data_dir}/training_folds/test_fold${fold}.tsv" \
  --out_file "${out_dir}/test.tsv" 

python '../../predict.py' \
  --checkpoint_file "${latest}" \
  --in_file "${out_dir}/test.tsv" \
  --out_file "${out_dir}/test.out" \
  --batch_size 32 \
  --gpus 1

  python '../../postprocess_sentence_selection_AuRED.py' \
    --in_file "${out_dir}/test.tsv" \
    --pred_sent_file "${out_dir}/test.out" \
    --pred_doc_file "${data_dir}/AuRED.json" \
    --out_file "${out_dir}/MLA_FEVER_zeroShot_fold${fold}.txt" \
    --out_json "${out_dir}/MLA_FEVER_zeroShot_fold${fold}.jsonl" \
    --max_evidence_per_claim 5
    # 
    
