#For each fold we set the lr to the one achieving the best MAP on dev set on the same fold.
fold=1
lr=2e-05
python predict_stance.py \
 --pretrained_model models/MARBERTv2_AuRED_stance_fold${fold}_${lr} \
 --data_file processed_AuRED/test_fold${fold}_full.txt \
 --lr ${lr} \
 --out_file CV_runs/MARBERTv2_AuRED_stance/MARBERTv2_AuRED_stance_fold${fold}_${lr}_testSet.txt
