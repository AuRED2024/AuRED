# dev set
# for fold in 1 2 3 4 5
# do 
#  for lr in 2e-5 3e-5 4e-5 5e-5
#  do
#   python verification_scorer.py \
#   --gold_file /data/release/AuRED_data.json \
#   --pred_file ./output/MARBERTv2/dev_lr${lr}_5epochs_fold${fold}.json \
#   --out_file  MARBERTv2_KGAT_verification_model_dev_results_DETAILED.csv
#  done
# done 

fold=5
lr=5e-5
python verification_scorer.py \
  --gold_file /data/release/AuRED_data.json \
  --pred_file ./output/MARBERTv2/test_lr${lr}_5epochs_fold${fold}.json \
  --out_file  MARBERTv2_KGAT_verification_model_test_results_DETAILED.csv