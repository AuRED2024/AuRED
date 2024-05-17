from src.contriever import Contriever
from transformers import AutoTokenizer
import pandas as pd
import re
import numpy as np
import pyarabic.araby as araby
import argparse
import json
import os
import sys
from sentence_transformers import util

def preprocess(text):
    cleaned_text = re.sub(r"http\S+", " ", text) # remove urls
    cleaned_text = re.sub(r"https\S+", " ", cleaned_text) # remove urls
    cleaned_text = re.sub(r"RT ", " ", cleaned_text) # remove rt
    cleaned_text= re.sub(r"@[\w]*", " ", cleaned_text) # remove handles
    cleaned_text=re.sub(r'[^0-9\u0600-\u06ff\u0750-\u077f\ufb50-\ufbc1\ufbd3-\ufd3f\ufd50-\ufd8f\ufd50-\ufd8f\ufe70-\ufefc\uFDF0-\uFDFD]+', ' ', cleaned_text)
    cleaned_text=araby.strip_diacritics(cleaned_text) 
    return cleaned_text

def get_tweet_id(tweet_link):
    return tweet_link.split("/")[-1]

def write_data(data,filename):
   with open(filename, 'a') as f:
    #    json.dump(data, f,indent=4,ensure_ascii = False)
       json.dump(data, f,ensure_ascii = False)
       f.write('\n')

def get_scores(claim_timeline,model,tokenizer):
    inputs = tokenizer(claim_timeline, padding=True, truncation=True, return_tensors="pt")
    embeddings = model(**inputs)
    print("done getting embedding")
    scores=[]
    for i in range(len(embeddings)-1):
        #score=embeddings[0] @ embeddings[i+1]
        score=util.cos_sim(embeddings[0],embeddings[i+1])
        scores.append(score.detach().numpy().item(0))
    # print(scores)
    return scores



def retrieve(infile,outfile,model,max_evidence):
    model = Contriever.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained("facebook/mcontriever") 

    with open(infile, 'r') as f:
         data = [json.loads(line) for line in f]
    output_claim_id=[]
    output_evidence_id=[]
    output_rank=[]
    output_score=[]

    for instance in data:
        claim_id=instance["id"]
        print(claim_id)
        claim=preprocess(instance["rumor"])
        timeline=[]
        timeline_ids=[]
        timeline_auth=[]
        for t in instance["timeline"]:
            timeline_auth.append(t[0])
            timeline_ids.append(t[1])
            timeline.append(preprocess(t[2]))
        if len(timeline)>100:
           scores=[]
           chunks = [timeline[x:x+100] for x in range(0, len(timeline), 100)]
           for chunk in chunks:
               claim_timeline=[]
               claim_timeline.append(claim)
               claim_timeline.extend(chunk)
               scores_chunk=get_scores(claim_timeline,model,tokenizer)
               scores.extend(scores_chunk)
           print("length of scores", len(scores))
           top_evidence_index=sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max_evidence]
           top_evidence_scores=sorted(scores,reverse=True)
           
    
        else:   
            claim_timeline=[]
            claim_timeline.append(claim)
            claim_timeline.extend(timeline)
            scores=get_scores(claim_timeline,model,tokenizer)
            top_evidence_index=sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max_evidence]
            top_evidence_scores=sorted(scores,reverse=True)

        counter=1
        for index in top_evidence_index:
            output_claim_id.append(claim_id)
            output_evidence_id.append(timeline_ids[index])
            output_rank.append(counter)
            counter=counter+1
            output_score.append(scores[index])
       
    output=pd.DataFrame()
    output['Rumor ID']=output_claim_id
    output['Q0']=['Q0']*len(output_claim_id)
    output['Evidence ID']=output_evidence_id
    output['Rank']=output_rank
    output['Score']=output_score
    output['Tag']=['mcontriever']*len(output_claim_id)
    print(output)
    output.to_csv(outfile,sep="\t",index=False,header=False)

   
if __name__ == "__main__":
      print("Dense retrieval using mContriever")
      parser = argparse.ArgumentParser()
      parser.add_argument('--infile')
      parser.add_argument('--outfile')
      parser.add_argument('--model')
      parser.add_argument('--max_evidence',default=5)
      args = parser.parse_args()
      retrieve(args.infile,args.outfile,args.model,args.max_evidence)
