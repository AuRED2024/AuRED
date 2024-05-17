data_dir='/data'
data_output='processed_AuRED'
#prepare dev and train data (only N negative examples)/we set neg_ratio=4
for fold in 1 2 3 4 5
do 
 for split in dev train
  do
   python 'preprocess_AuRED.py' \
    --in_file "${data_dir}/AuRED.json" \
    --neg_ratio 4 \
    --fold_file "${data_dir}/training_folds/${split}_fold${fold}.tsv" \
    --out_file "${data_output}/${split}_fold${fold}.txt" \
    --training
  done
done

#dev and test data for prediction for full data without choosing random negatives
for fold in 1 2 3 4 5
do 
 for split in dev test
  do
   python 'preprocess_AuRED.py' \
    --in_file "${data_dir}/AuRED.json" \
    --fold_file "${data_dir}/training_folds/${split}_fold${fold}.tsv" \
    --out_file "${data_output}/${split}_fold${fold}_full.txt" \
  done
done

#For AURED*, we just need the test set for prediction
data_dir='/data'
data_output='processed_AuRED/AuRED_star'
split='test'
for fold in 1 2 3 4 5
 do 
   python 'preprocess_AuRED.py' \
   --in_file "${data_dir}/AuRED_star.json" \
   --fold_file "${data_dir}/training_folds/${split}_fold${fold}.tsv" \
   --out_file "${data_output}/${split}_fold${fold}_full.txt" \
 done
