#For AuRED*, we just used the model fine-tuned on AuRED to retrieve and verify so here we are just formatting the retrieved data on the test set to predict verification label.
epochs_num=5
fold=5
lr=5e-5 #[2e-5,2e-5,4e-5,2e-5,5e-5]


    
python test.py  --outdir ./output/CV_experiments/verification_data \
    --trec_outdir ./trec_output/CV_experiments/ \
    --test_path ../data/testing_format/AuRED_star/all_AuRED_test_fold${fold}.json \
    --bert_pretrain ../MARBERTv2_base \
    --checkpoint ../checkpoint/retrieval_model/CV_experiments/model.best_lr${lr}_${epochs_num}epochs_fold${fold}.pt \
    --name test_lr${lr}_${epochs_num}epochs_fold${fold}_GoldSet.json \
    --trec_name test_lr${lr}_${epochs_num}epochs_fold${fold}_GoldSet.txt \
    --batch_size 16 


python process_data.py --retrieval_file /retrieval_model/output/CV_experiments/verification_data/test_lr${lr}_${epochs_num}epochs_fold${fold}_GoldSet.json --gold_file /data/testing_format/AuRED*/all_AuRED_test_fold${fold}.json --output ./output/CV_experiments/verification_data/bert_test_lr${lr}_${epochs_num}epochs_fold${fold}_GoldSet.json --test
