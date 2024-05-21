for fold in 1 2 3 4 5
do
 python test.py  --outdir ./output/CV_experiments/KGAT_zeroShot/verification_data \
    --trec_outdir ./trec_output/CV_experiments/KGAT_zeroShot \
    --test_path ../data/testing_format/all_AuRED_test_fold${fold}.json \
    --bert_pretrain ../bert-base-multilingual-uncased \
    --checkpoint ../checkpoint/KGAT_zeroShot_retrieval_model/model.best.pt \
    --name test_fold${fold}.json \
    --trec_name test_fold${fold}.txt \
    --batch_size 32
