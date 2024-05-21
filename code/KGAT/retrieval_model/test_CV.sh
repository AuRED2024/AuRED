# #dev sets
for fold in 1 2 3 4 5
do 
 for epochs_num in 5
  do
   for lr in 2e-5 3e-5 4e-5 5e-5
    do
    python test.py  --outdir ./output/CV_experiments/ \
    --trec_outdir ./trec_output/CV_experiments/ \
    --test_path ../data/testing_format/all_AuRED_dev_fold${fold}.json \
    --bert_pretrain ../MARBERTv2_base \
    --checkpoint ../checkpoint/retrieval_model/CV_experiments/model.best_lr${lr}_${epochs_num}epochs_fold${fold}.pt \
    --name dev.json \
    --trec_name lr${lr}_${epochs_num}epochs_fold${fold}.txt \
    --batch_size 16 
   done
  done
done

#test on test sets/based on best learning rates on the dev set above (best MAP)
fold=1
lr=2e-5
epochs_num=5

python test.py  --outdir ./output/CV_experiments/ \
    --trec_outdir ./trec_output/CV_experiments/ \
    --test_path ../data/testing_format/all_AuRED_test_fold${fold}.json \
    --bert_pretrain ../MARBERTv2_base \
    --checkpoint ../checkpoint/retrieval_model/CV_experiments/model.best_lr${lr}_${epochs_num}epochs_fold${fold}.pt \
    --name test.json \
    --trec_name lr${lr}_${epochs_num}epochs_fold${fold}_testSet.txt \
    --batch_size 16 
fold=2
lr=2e-5
epochs_num=5

python test.py  --outdir ./output/CV_experiments/ \
    --trec_outdir ./trec_output/CV_experiments/ \
    --test_path ../data/testing_format/all_AuRED_test_fold${fold}.json \
    --bert_pretrain ../MARBERTv2_base \
    --checkpoint ../checkpoint/retrieval_model/CV_experiments/model.best_lr${lr}_${epochs_num}epochs_fold${fold}.pt \
    --name test.json \
    --trec_name lr${lr}_${epochs_num}epochs_fold${fold}_testSet.txt \
    --batch_size 16 

fold=3
lr=4e-5
epochs_num=5

python test.py  --outdir ./output/CV_experiments/ \
    --trec_outdir ./trec_output/CV_experiments/ \
    --test_path ../data/testing_format/all_AuRED_test_fold${fold}.json \
    --bert_pretrain ../MARBERTv2_base \
    --checkpoint ../checkpoint/retrieval_model/CV_experiments/model.best_lr${lr}_${epochs_num}epochs_fold${fold}.pt \
    --name test.json \
    --trec_name lr${lr}_${epochs_num}epochs_fold${fold}_testSet.txt \
    --batch_size 16 

fold=4
lr=2e-5
epochs_num=5

python test.py  --outdir ./output/CV_experiments/ \
    --trec_outdir ./trec_output/CV_experiments/ \
    --test_path ../data/testing_format/all_AuRED_test_fold${fold}.json \
    --bert_pretrain ../MARBERTv2_base \
    --checkpoint ../checkpoint/retrieval_model/CV_experiments/model.best_lr${lr}_${epochs_num}epochs_fold${fold}.pt \
    --name test.json \
    --trec_name lr${lr}_${epochs_num}epochs_fold${fold}_testSet.txt \
    --batch_size 16 

fold=5
lr=5e-5
epochs_num=5

python test.py  --outdir ./output/CV_experiments/ \
    --trec_outdir ./trec_output/CV_experiments/ \
    --test_path ../data/testing_format/all_AuRED_test_fold${fold}.json \
    --bert_pretrain ../MARBERTv2_base \
    --checkpoint ../checkpoint/retrieval_model/CV_experiments/model.best_lr${lr}_${epochs_num}epochs_fold${fold}.pt \
    --name test.json \
    --trec_name lr${lr}_${epochs_num}epochs_fold${fold}_testSet.txt \
    --batch_size 16 
