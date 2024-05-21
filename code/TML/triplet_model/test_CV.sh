##dev sets
for fold in 1 2 3 4 5
 do 
  for epochs_num in 5
   do
    for lr in 2e-5 3e-5 4e-5 5e-5
     do
     python test.py  --outdir ./output/CV_experiments/ \
     --test_path ../data/testing_format/all_AuRED_dev_fold${fold}.json \
     --bert_pretrain ../MARBERTv2_base \
     --checkpoint ../checkpoint/CV_experiments/triplet_model/lr${lr}_${epochs_num}epochs_fold${fold}/model.best.pt \
     --name dev.json \
     --trec_name lr${lr}_${epochs_num}epochs_fold${fold}.txt \
     --batch_size 32 \
     --gpu 0 
    done
   done
 done




epochs_num=5
fold=5
lr=5e-5  #[2e-5,2e-5,3e-5,2e-5,5e-5]
python test.py  --outdir ./output/CV_experiments/ \
    --test_path ../data/testing_format/all_AuRED_test_fold${fold}.json \
    --bert_pretrain ../MARBERTv2_base \
    --checkpoint ../checkpoint/CV_experiments/triplet_model/lr${lr}_${epochs_num}epochs_fold${fold}/model.best.pt \
    --name test.json \
    --trec_name lr${lr}_${epochs_num}epochs_fold${fold}_testSet.txt \
    --batch_size 8 \
    --gpu 0 


