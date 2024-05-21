## Cross-lingual Zero-shot experiments
1. Use the original KGAT code [here](https://github.com/thunlp/KernelGAT/tree/master/retrieval_model) where they fine-tuned the model using FEVER English dataset. However update the train.sh as shown below:

> python train.py --outdir ../checkpoint/retrieval_model \
--train_path ../data/train_pair \
--valid_path ../data/dev_pair \
--bert_pretrain ../bert-base-multilingual-uncased </br>

2. Use the fine-tuned model to retrieve evidence for AuRED data. 

> bash test_CV_zeroShot.sh </br>

## In-domain Fine-tuning Scenario

