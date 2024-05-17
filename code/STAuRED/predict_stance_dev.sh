#MARBERTv2_AuRED_stance
for fold in 1 2 3 4 5
  do
   for lr in 2e-05 3e-05 4e-05 5e-05
    do
     python predict_stance.py \
     --pretrained_model models/MARBERTv2_AuRED_stance_fold${fold}_${lr} \
     --data_file processed_AuRED/dev_fold${fold}_full.txt \
     --lr ${lr} \
     --out_file CV_runs/MARBERTv2_AuRED_stance/MARBERTv2_AuRED_stance_fold${fold}_${lr}.txt 
   done
done
