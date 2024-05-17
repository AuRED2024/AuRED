for fold in 1 2 3 4 5
do 
 for epochs_num in 5
  do
   for lr in 2e-5 3e-5 4e-5 5e-5
    do 

     python train.py --outdir ../checkpoint/retrieval_model/CV_experiments \
     --train_path ../data/training_folds_pairs/train_fold${fold}_pair \
     --valid_path ../data/training_folds_pairs/dev_fold${fold}_pair \
     --bert_pretrain ../MARBERTv2_base \
     --setup "lr${lr}_${epochs_num}epochs_fold${fold}" \
     --train_batch_size 8 \
     --learning_rate "${lr}" \
     --num_train_epochs "${epochs_num}" \
     --eval_step 750
    done
  done
done
