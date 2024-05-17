import json
import pandas as pd
import argparse
import re
import pyarabic.araby as araby
import os
from pyserini.search.lucene import LuceneSearcher

def normalize(text):
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    return(text)

def preprocess(text):
    cleaned_text = re.sub(r"http\S+", " ", text) # remove urls
    cleaned_text = re.sub(r"https\S+", " ", cleaned_text) # remove urls
    cleaned_text = re.sub(r"RT ", " ", cleaned_text) # remove rt
    cleaned_text= re.sub(r"@[\w]*", " ", cleaned_text) # remove handles
    cleaned_text=re.sub(r'[^0-9\u0600-\u06ff\u0750-\u077f\ufb50-\ufbc1\ufbd3-\ufd3f\ufd50-\ufd8f\ufd50-\ufd8f\ufe70-\ufefc\uFDF0-\uFDFD]+', ' ', cleaned_text)
    cleaned_text=araby.strip_diacritics(cleaned_text)
    return cleaned_text

def BM25(data_path,saving_file,K=5):
    with open(data_path, 'r') as f:
         data = [json.loads(line) for line in f]
    finaldf=pd.DataFrame()
    claim_id_arr=[]
    Q0=[]
    evidence_id=[]
    rank=[]
    score=[]
    tag=[]
    for instance in data: 
        claim_id=instance["id"]
        print(claim_id)
        query=normalize(preprocess(instance["rumor"]))
        searcher = LuceneSearcher("./AuRED_indexes/%s_index"%claim_id)
        searcher.set_language('ar')
        #hits = searcher.search(query,k=K)
        hits = searcher.search(query,k=5)
        for i in range(len(hits)):
            claim_id_arr.append(claim_id)
            tag.append('BM25')
            evidence_id.append(hits[i].docid)
            score.append(hits[i].score)
            rank.append(i+1)


    finaldf['claimID']= claim_id_arr
    finaldf['Q0']=['Q0']*len( claim_id_arr)
    finaldf['evidenceID']=evidence_id
    finaldf['rank']=rank
    finaldf['score']=score
    finaldf['tag']=tag
    finaldf.to_csv('./runs/%s'%saving_file,sep='\t',index=False,header=False)
  
    
if __name__ == "__main__":
    print("retrieving using pyserini")
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile')
    parser.add_argument('--outfile')
    #parser.add_argument('--split',default="dev")
    parser.add_argument('--k',default=5)
    args = parser.parse_args()
    BM25(args.infile,args.outfile,args.k)

