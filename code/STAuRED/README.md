### To fine-tune STAuRED model you need to follow the below steps:
1. Prepare data by running the following:
  >bash preprocess_AuRED.sh<\br>

**Note: we shared the output of this step [here](https://github.com/AuRED2024/AuRED/tree/main/code/STAuRED/processed_AuRED). You can use our prepared data where we selected 4 negative examples. If you want to experiement with different numbers of negative examples, you need to modify *neg_ratio* parameter to the required value in *preprocess_AuRED.sh* **

2. To start fine-tuning the model run the below:
 >python train.py<\br>

3. To retrieve evidence for the dev sets using the fine-tuned models run the below:
  >bash predict_stance_dev.sh<\br>
 
