#!/bin/bash
#
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Authors: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.
#
#SBATCH --job-name=predict_bert-base
#SBATCH --out='predict_bert-base.log'
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:tesla_a100:1

conda_setup="/home/smg/$(whoami)/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${conda_setup}" ]]; then
  #shellcheck disable=SC1090
  . "${conda_setup}"
  conda activate mla
fi

set -ex
data_dir='/data/'
pretrained='UBC-NLP/MARBERTv2'
max_len=128
for fold in 1 2 3 4 5
do 
 for epochs_num in 5
  do
   for lr in 2e-5 3e-5 4e-5 5e-5
    do
     model_dir="CV_experiments/MARBERTv2-${max_len}-mod_4neg_lr${lr}_${epochs_num}epochs_fold${fold}"
     out_dir="CV_experiments/MARBERTv2-${max_len}-out_4neg_lr${lr}_${epochs_num}epochs_fold${fold}"

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
  
     python '../../preprocess_sentence_selection_AuRED_CV.py' \
     --in_file "${data_dir}/AuRED.json" \
     --fold_file "${data_dir}/training_folds/dev_fold${fold}.tsv" \
     --out_file "${out_dir}/dev.tsv"

     python '../../predict.py' \
     --checkpoint_file "${latest}" \
     --in_file "${out_dir}/dev.tsv" \
     --out_file "${out_dir}/dev.out" \
     --batch_size 256 \
     --gpus 1
  

     python '../../postprocess_sentence_selection_AuRED.py' \
     --in_file "${out_dir}/dev.tsv" \
     --pred_sent_file "${out_dir}/dev.out" \
     --pred_doc_file "${data_dir}/AuRED.json" \
     --out_file "${out_dir}/MLA_dev_4neg_lr${lr}_${epochs_num}epochs_fold${fold}.txt" \
     --out_json "${out_dir}/MLA_dev_4neg_lr${lr}_${epochs_num}epochs_fold${fold}.jsonl" \
     --max_evidence_per_claim 5
     
     #training data prediction to be used for claim verification
     python '../../preprocess_sentence_selection_AuRED_CV.py' \
     --in_file "${data_dir}/AuRED.json" \
     --fold_file "${data_dir}/training_folds/train_fold${fold}.tsv" \
     --out_file "${out_dir}/train.tsv"

     python '../../predict.py' \
     --checkpoint_file "${latest}" \
     --in_file "${out_dir}/train.tsv" \
     --out_file "${out_dir}/train.out" \
     --batch_size 256 \
     --gpus 1
  
     python '../../postprocess_sentence_selection_AuRED.py' \
     --in_file "${out_dir}/train.tsv" \
     --pred_sent_file "${out_dir}/train.out" \
     --pred_doc_file "${data_dir}/AuRED.json" \
     --out_file "${out_dir}/MLA_train_4neg_lr${lr}_${epochs_num}epochs_fold${fold}.txt" \
     --out_json "${out_dir}/MLA_train_4neg_lr${lr}_${epochs_num}epochs_fold${fold}.jsonl" \
     --max_evidence_per_claim 5
  
    done
  done
done
