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
import time
start_time = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device,flush=True)

print("done importing",flush=True)

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
from transformers import BertTokenizer

    

class Bert(Dataset):

  def __init__(self, train_df, val_df):
    #0 is other, 1 is refutes, and 2 is supports
    self.label_dict = {0:0,1:1,2:2}

    self.train_df = train_df
    self.val_df = val_df


    self.base_path = '/content/'
    self.tokenizer = BertTokenizer.from_pretrained("UBC-NLP/MARBERTv2")
    self.train_data = None
    self.val_data = None
    self.init_data()

  def init_data(self):
    self.train_data = self.load_data(self.train_df)
    self.val_data = self.load_data(self.val_df)
    

  def load_data(self, df):
    MAX_LEN = 512
    token_ids = []
    mask_ids = []
    seg_ids = []
    y = []

    premise_list = df['claim'].to_list()
    hypothesis_list = df['authority_tweet'].to_list()
    label_list = df['label'].to_list()

    for (premise, hypothesis, label) in zip(premise_list, hypothesis_list, label_list):
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
      y.append(self.label_dict[label])
    
    token_ids = pad_sequence(token_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    seg_ids = pad_sequence(seg_ids, batch_first=True)
    y = torch.tensor(y)
    dataset = TensorDataset(token_ids, mask_ids, seg_ids, y)
    print(len(dataset),flush=True)
    return dataset

  def get_data_loaders(self, batch_size=16, shuffle=True):
    train_loader = DataLoader(
      self.train_data,
      shuffle=shuffle,
      batch_size=batch_size
    )

    val_loader = DataLoader(
      self.val_data,
      shuffle=shuffle,
      batch_size=batch_size
    )
  

    return train_loader, val_loader

#calculate accuracy
def multi_acc(y_pred, y_test):
  acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_test.size(0))
  return acc


def train(model, train_loader, val_loader, optimizer):
  #early stopping
  last_loss = 100
  #patience = 5
  triggertimes = 0  

  total_step = len(train_loader)

  for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    total_train_loss = 0
    total_train_acc  = 0
    for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(train_loader):
      optimizer.zero_grad()
      pair_token_ids = pair_token_ids.to(device)
      mask_ids = mask_ids.to(device)
      seg_ids = seg_ids.to(device)
      labels = y.to(device)
      loss, prediction = model(pair_token_ids, 
                             token_type_ids=seg_ids, 
                             attention_mask=mask_ids, 
                             labels=labels).values()

      acc = multi_acc(prediction, labels)
      loss.backward()
      optimizer.step()
      
      total_train_loss += loss.item()
      total_train_acc  += acc.item()
      
    train_acc  = total_train_acc/len(train_loader)
    train_loss = total_train_loss/len(train_loader)
    model.eval()
    total_val_acc  = 0
    total_val_loss = 0
    with torch.no_grad():
      for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(val_loader):
        optimizer.zero_grad()
        pair_token_ids = pair_token_ids.to(device)
        mask_ids = mask_ids.to(device)
        seg_ids = seg_ids.to(device)
        labels = y.to(device)

        loss, prediction = model(pair_token_ids, 
                             token_type_ids=seg_ids, 
                             attention_mask=mask_ids, 
                             labels=labels).values()
        
        acc = multi_acc(prediction, labels)
        
        total_val_loss += loss.item()
        total_val_acc  += acc.item()

    val_acc  = total_val_acc/len(val_loader)
    val_loss = total_val_loss/len(val_loader)

    last_loss = val_loss
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}',flush=True)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds),flush=True)
    print("--- %s seconds ---" % (time.time() - start_time),flush=True)
  return model

#=================================================================================================================================================================
EPOCHS=5
seed=2023
learning_rates=[2e-5,3e-5,4e-5,5e-5]
torch.manual_seed(seed)
results_file="MARBERTv2_AuRED_stance.txt"
previous_model="UBC-NLP/MARBERTv2"
folds=[1,2,3,4,5]
for fold in folds:            
 train_df=pd.read_csv("processed_AuRED/train_fold%s.txt"%fold,sep="\t",names=['claim_id',"claim","auth","auth_tweet_id","authority_tweet","label"])
 val_df=pd.read_csv("processed_AuRED/dev_fold%s.txt"%fold,sep="\t",names=['claim_id',"claim","auth","auth_tweet_id","authority_tweet","label"])
 train_df=train_df[["claim","authority_tweet","label"]] 
 val_df=val_df[["claim","authority_tweet","label"]]       
 train_df=train_df.dropna().reset_index(drop=True)
 print("size of train",len(train_df))
 val_df=val_df.dropna().reset_index(drop=True)
 print("size of dev",len(val_df))
 print("Done reading data",flush=True)
 print(train_df.head())
 print(val_df.head()) 
  
 for learning_rate in learning_rates: 
   model = BertForSequenceClassification.from_pretrained(previous_model, num_labels=3)
   model.to(device)   
   output_dir="models/MARBERTv2_AuRED_fold%s_%s"%(fold,learning_rate)
   param_optimizer = list(model.named_parameters())
   pretrained = model.bert.parameters()
   # Get names of pretrained parameters (including `bert.` prefix)
   pretrained_names = [f'bert.{k}' for (k, v) in model.bert.named_parameters()]
   new_params= [v for k, v in model.named_parameters() if k not in pretrained_names]
   optimizer = AdamW([{'params': pretrained}, {'params': new_params, 'lr': learning_rate * 10}], lr=learning_rate)     
   dataset = Bert(train_df, val_df)
   train_loader, val_loader = dataset.get_data_loaders(batch_size=8)
   print("Done processing data and will start training......",flush=True)
   model=train(model, train_loader, val_loader, optimizer)
   
   if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    model.save_pretrained(output_dir)
     
   finetuned_model = BertForSequenceClassification.from_pretrained(output_dir, num_labels=3) 
   finetuned_model.to(device)
   probs_all = []
   golden=[]
   with torch.no_grad():
    for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(val_loader):
                    optimizer.zero_grad()
                    pair_token_ids = pair_token_ids.to(device)
                    mask_ids = mask_ids.to(device)
                    seg_ids = seg_ids.to(device)
                    labels = y.to(device)

                    _, prediction = finetuned_model(pair_token_ids, 
                             token_type_ids=seg_ids, 
                             attention_mask=mask_ids, 
                             labels=labels).values()
        
                    probs_all += prediction.tolist()
                    golden+=labels.tolist()
   F1_score=f1_score(golden,torch.log_softmax(torch.tensor(probs_all), dim=1).argmax(dim=1).tolist(),average='macro')
   print("F1_score",F1_score,flush=True)
   print("*****************************************************************************************",flush=True)
   result_list=[fold,learning_rate,F1_score]
   with open("results/%s"%results_file, 'a') as f_object:
    writer_object = writer(f_object,delimiter ="\t")
    writer_object.writerow(result_list)
    f_object.close()
