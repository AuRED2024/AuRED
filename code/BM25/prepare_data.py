import json
import pandas as pd
import argparse
import re
import pyarabic.araby as araby
import os
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

def process_data(data_path):
    with open(data_path, 'r') as f:
         data = [json.loads(line) for line in f]
    
    for instance in data:
        claim_id=instance["id"]
        print(claim_id)
        print(len(instance["timeline"]))
        evidence_text=[]
        evidence_ids=[]
        for t in instance["timeline"]:
            evidence_ids.append(t[1])
            print(t[1])
            evidence_text.append(normalize(preprocess(t[2])))
        # print(evidence_text)
        claim_data=pd.DataFrame()
        claim_data['id']=evidence_ids
        claim_data['contents']=evidence_text
        result = claim_data.to_json(orient="records",force_ascii=False)
        path="AuRED_JSON/%s"%claim_id
        isExist = os.path.exists(path)
        if not isExist:
           # Create a new directory because it does not exist
           os.makedirs(path)
           print("The new directory is created!")
        with open(path+"/%s.json"%claim_id, "w",encoding="utf-8") as outfile:
             outfile.write(result)
    

 

if __name__ == "__main__":
    print("preparing data for indexing using pyserini")
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile') 
    args = parser.parse_args()
    process_data(args.infile)
