import pandas as pd
import re
import torch
import random
from torch.utils.data import Dataset, TensorDataset, DataLoader, SequentialSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
from transformers import BertForSequenceClassification, AdamW
import time
import sys
from csv import writer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import argparse
# print(device,flush=True)

# print("done importing",flush=True)

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
from transformers import BertTokenizer

class Bert(Dataset):

  def __init__(self, val_df):
   #  self.label_dict = {1:1, 0:0}
    self.val_df = val_df


    self.base_path = '/content/'
    self.tokenizer = BertTokenizer.from_pretrained('UBC-NLP/MARBERTv2')
    self.val_data = None
    self.init_data()

  def init_data(self):
    self.val_data = self.load_data(self.val_df)
    

  def load_data(self, df):
    MAX_LEN = 512
    token_ids = []
    mask_ids = []
    seg_ids = []
    y = []

    premise_list = df['claim'].to_list()
    hypothesis_list = df['authority_tweet'].to_list()
    #label_list = df['label'].to_list()

    for (premise, hypothesis) in zip(premise_list, hypothesis_list):
      premise_id = self.tokenizer.encode(premise, add_special_tokens = False)
      hypothesis_id = self.tokenizer.encode(hypothesis, add_special_tokens = False, max_length=512-3-len(premise_id), truncation=True)
      pair_token_ids = [self.tokenizer.cls_token_id] + premise_id + [self.tokenizer.sep_token_id] + hypothesis_id + [self.tokenizer.sep_token_id]
      premise_len = len(premise_id)
      hypothesis_len = len(hypothesis_id)

      segment_ids = torch.tensor([0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
      attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values

      token_ids.append(torch.tensor(pair_token_ids))
      seg_ids.append(segment_ids)
      mask_ids.append(attention_mask_ids)
      # y.append(self.label_dict[label])
    
    token_ids = pad_sequence(token_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    seg_ids = pad_sequence(seg_ids, batch_first=True)
    #y = torch.tensor(y)
    dataset = TensorDataset(token_ids, mask_ids, seg_ids)
    print(len(dataset),flush=True)
    return dataset

  def get_data_loaders(self, batch_size=16, shuffle=False):
  
    val_loader = DataLoader(
      self.val_data,
      shuffle=shuffle,
      batch_size=batch_size
    )

    return val_loader

#calculate accuracy
def multi_acc(y_pred, y_test):
  acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_test.size(0))
  return acc





def build_args():
    parser = argparse.ArgumentParser()
   #  parser.add_argument("--fold", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--max_evidence_per_claim", type=int, default=5)
    parser.add_argument("--min_score", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    return parser.parse_args()


def main():
    
    args = build_args()
   
    learning_rate=args.lr
    finetuned_model = BertForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=3) 
    finetuned_model.to(device)
    pretrained =  finetuned_model.bert.parameters()
    # Get names of pretrained parameters (including `bert.` prefix)
    pretrained_names = [f'bert.{k}' for (k, v) in  finetuned_model.bert.named_parameters()]
    new_params= [v for k, v in  finetuned_model.named_parameters() if k not in pretrained_names]
    optimizer = AdamW([{'params': pretrained}, {'params': new_params, 'lr': learning_rate * 10}], lr= learning_rate)
    probs_all = []
    golden=[]

    val_df=pd.read_csv(args.data_file,sep="\t",names=['claim_id',"claim","auth","auth_tweet_id","authority_tweet"])
    val_df=val_df.dropna().reset_index(drop=True)
    val_df=val_df[val_df['authority_tweet']!=" "].reset_index(drop=True)
    print(val_df.head())
    print(len(val_df))
    dataset = Bert(val_df)
    val_loader = dataset.get_data_loaders(batch_size=16)
    
    with torch.no_grad():
     for batch_idx, (pair_token_ids, mask_ids, seg_ids) in enumerate(val_loader):
                    optimizer.zero_grad()
                    pair_token_ids = pair_token_ids.to(device)
                    mask_ids = mask_ids.to(device)
                    seg_ids = seg_ids.to(device)
                    #labels = y.to(device)

                    output= finetuned_model(pair_token_ids, 
                             token_type_ids=seg_ids, 
                             attention_mask=mask_ids)
                    logits=output.logits
                    probabilities=torch.softmax(torch.tensor(logits), dim=1).tolist()
                    for instance in probabilities:
                        probs_all.append(1-instance[0])
   #  print(len(probs_all))

   
    temp=pd.DataFrame()
    final_df=pd.DataFrame()
    temp['claim_id']=val_df['claim_id'].tolist()
    temp['Q0']=['Q0']*len(val_df['claim_id'].tolist())
    temp["auth_tweet_id"]=val_df['auth_tweet_id'].tolist()
    
    temp["score"]=probs_all
    temp["tag"]=["MARBERTv2-Stance"]*len(val_df['claim_id'].tolist())
    final_df=temp.groupby(['claim_id']).apply(lambda x: x.sort_values(['score'], ascending=False)[0:args.max_evidence_per_claim])
    final_df["rank"]=temp.groupby(['claim_id']).apply(lambda x: x.sort_values(['score'], ascending=False)[0:args.max_evidence_per_claim]['score'].rank(method="first", ascending=False)).astype(int)
    final_df=final_df[["claim_id","Q0","auth_tweet_id","rank","score","tag"]]
    final_df.to_csv(args.out_file,sep="\t",index=False,header=False)
    
if __name__ == "__main__":
    main()
