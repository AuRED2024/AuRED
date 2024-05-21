epochs_num=5
fold=5
lr=5e-5 #[2e-5,2e-5,3e-5,2e-5,5e-5]
python test.py  --outdir ./output/CV_experiments/ \
    --test_path ../data/testing_format/AuRED_star/all_AuRED_test_fold${fold}.json \
    --bert_pretrain ../MARBERTv2_base \
    --checkpoint ../checkpoint/CV_experiments/triplet_model/lr${lr}_${epochs_num}epochs_fold${fold}/model.best.pt \
    --name test_gold.json \
    --trec_name lr${lr}_${epochs_num}epochs_fold${fold}_testSet_GoldSet.txt \
    --batch_size 32 \
    --gpu 0 