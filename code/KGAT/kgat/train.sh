fold=5
pred_lr=5e-5 #best learning rate for evidence retrieval for the specific fold
for lr in 2e-5 3e-5 4e-5 5e-5
do
 python train.py --outdir ../checkpoint/kgat/MARBERTv2/lr${lr}_5epochs_fold${fold} \
 --train_path /retrieval_model/output/CV_experiments/verification_data/bert_train_lr${pred_lr}_5epochs_fold${fold}.json \
 --valid_path /retrieval_model/output/CV_experiments/verification_data/bert_dev_lr${pred_lr}_5epochs_fold${fold}.json \
 --bert_pretrain ../MARBERTv2_base \
 --eval_step 1 \
 --learning_rate ${lr} \
 --num_train_epochs 5
done
