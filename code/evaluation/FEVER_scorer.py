import json
import sys
import argparse
import pandas as pd


def instance_macro_precision(actual_evidence,predicted_evidence, max_evidence=None):
    this_precision = 0.0
    this_precision_hits = 0.0
    #check if the claim has NOT ENOUGH INFO (no evidence)
    if actual_evidence[0]!=-1:
        for prediction in predicted_evidence:
            if prediction in actual_evidence:
                this_precision += 1.0
            this_precision_hits += 1.0

        return (this_precision / this_precision_hits) if this_precision_hits > 0 else 1.0, 1.0

    return 0.0, 0.0

def instance_macro_recall(actual_evidence,predicted_evidence, max_evidence=None):
    # We only want to score F1/Precision/Recall of recalled evidence for NEI claims
    #check if the claim has NOT ENOUGH INFO (no evidence)
    if actual_evidence[0]!=-1:
        for prediction in predicted_evidence:
            if prediction in actual_evidence:
                return 1.0, 1.0
        return 0.0, 1.0
    return 0.0, 0.0



def compute_recall(actual,predicted, max_evidence=None):
    actual_data=pd.read_csv(actual,sep="\t",names=['claim_id','0',"evidence_id","relevance"])
    predicted_data =pd.read_csv(predicted,sep="\t",names=['claim_id','Q0',"evidence_id","rank","score","tag"])
  
    actual=actual_data.groupby(['claim_id'])
    predicted=predicted_data.groupby(['claim_id'])

    macro_recall = 0
    macro_recall_hits = 0

    for index,instance in predicted:
        claim_id=index
        predicted_evidence=instance['evidence_id'].tolist()
        actual_evidence=actual_data[actual_data['claim_id']==claim_id]['evidence_id'].reset_index(drop=True).tolist()
        macro_rec = instance_macro_recall(actual_evidence, predicted_evidence, max_evidence)
        macro_recall += macro_rec[0]
        macro_recall_hits += macro_rec[1]
   
    rec = (macro_recall / macro_recall_hits) if macro_recall_hits > 0 else 0.0
    return rec

def compute_macro_precision(actual,predicted, max_evidence=None):
 
    actual_data=pd.read_csv(actual,sep="\t",names=['claim_id','0',"evidence_id","relevance"])
    predicted_data =pd.read_csv(predicted,sep="\t",names=['claim_id','Q0',"evidence_id","rank","score","tag"])

    actual=actual_data.groupby(['claim_id'])
    predicted=predicted_data.groupby(['claim_id'])
    macro_precision = 0
    macro_precision_hits = 0                          
    for index,instance in predicted:
        claim_id=index
        predicted_evidence=instance['evidence_id'].tolist()
        actual_evidence=actual_data[actual_data['claim_id']==claim_id]['evidence_id'].reset_index(drop=True).tolist()
        macro_prec = instance_macro_precision(actual_evidence, predicted_evidence, max_evidence)
        macro_precision += macro_prec[0]
        macro_precision_hits += macro_prec[1]
 
    pr = (macro_precision / macro_precision_hits) if macro_precision_hits > 0 else 1.0
    return pr
if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument('--actual')
      parser.add_argument('--predicted')
      parser.add_argument('--max_evidence',default=5)
      args = parser.parse_args()
      actual_path=args.actual
      predicted_path=args.predicted
      prec=compute_macro_precision(actual_path,predicted_path)
      print(prec)
      print("Precision: ",round(prec,3))
      rec=compute_recall(actual_path,predicted_path)
      print("Recall: ",round(rec,3))
      f1 = 2.0 * prec * rec / (prec + rec)
      print("F1",round(f1,3))






