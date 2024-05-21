#This is to be done to all folds based on best learning rate on dev set 
fold=5 #[1,2,3,4,5]
epochs_num=5
lr=5e-5 #[2e-5,2e-5,4e-5,2e-5,5e-5]

python test.py  --outdir ./output/CV_experiments/verification_data \
    --trec_outdir ./trec_output/CV_experiments/ \
    --test_path ../data/testing_format/all_AuRED_train_fold${fold}.json \
    --bert_pretrain ../MARBERTv2_base \
    --checkpoint ../checkpoint/retrieval_model/CV_experiments/model.best_lr${lr}_${epochs_num}epochs_fold${fold}.pt \
    --name train_lr${lr}_${epochs_num}epochs_fold${fold}.json \
    --trec_name train_lr${lr}_${epochs_num}epochs_fold${fold}.txt \
    --batch_size 16 

python test.py  --outdir ./output/CV_experiments/verification_data \
    --trec_outdir ./trec_output/CV_experiments/ \
    --test_path ../data/testing_format/all_AuRED_dev_fold${fold}.json \
    --bert_pretrain ../MARBERTv2_base \
    --checkpoint ../checkpoint/retrieval_model/CV_experiments/model.best_lr${lr}_${epochs_num}epochs_fold${fold}.pt \
    --name dev_lr${lr}_${epochs_num}epochs_fold${fold}.json \
    --trec_name dev_lr${lr}_${epochs_num}epochs_fold${fold}.txt \
    --batch_size 16
    
python test.py  --outdir ./output/CV_experiments/verification_data \
    --trec_outdir ./trec_output/CV_experiments/ \
    --test_path ../data/testing_format/all_AuRED_test_fold${fold}.json \
    --bert_pretrain ../MARBERTv2_base \
    --checkpoint ../checkpoint/retrieval_model/CV_experiments/model.best_lr${lr}_${epochs_num}epochs_fold${fold}.pt \
    --name test_lr${lr}_${epochs_num}epochs_fold${fold}.json \
    --trec_name test_lr${lr}_${epochs_num}epochs_fold${fold}.txt \
    --batch_size 16 

python process_data.py --retrieval_file /retrieval_model/output/CV_experiments/verification_data/train_lr${lr}_${epochs_num}epochs_fold${fold}.json --gold_file /data/testing_format/all_AuRED_train_fold${fold}.json --output ./output/CV_experiments/verification_data/bert_train_lr${lr}_${epochs_num}epochs_fold${fold}.json
python process_data.py --retrieval_file /retrieval_model/output/CV_experiments/verification_data/dev_lr${lr}_${epochs_num}epochs_fold${fold}.json --gold_file /data/testing_format/all_AuRED_dev_fold${fold}.json --output ./output/CV_experiments/verification_data/bert_dev_lr${lr}_${epochs_num}epochs_fold${fold}.json
python process_data.py --retrieval_file /retrieval_model/output/CV_experiments/verification_data/dev_lr${lr}_${epochs_num}epochs_fold${fold}.json --gold_file /data/testing_format/all_AuRED_dev_fold${fold}.json --output ./output/CV_experiments/verification_data/bert_eval_lr${lr}_${epochs_num}epochs_fold${fold}.json --test
python process_data.py --retrieval_file /retrieval_model/output/CV_experiments/verification_data/test_lr${lr}_${epochs_num}epochs_fold${fold}.json --gold_file /data/testing_format/all_AuRED_test_fold${fold}.json --output ./output/CV_experiments/verification_data/bert_test_lr${lr}_${epochs_num}epochs_fold${fold}.json --test
