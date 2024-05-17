import argparse
import jsonlines
import random
import io
from tqdm import tqdm
from collections import defaultdict
import re
import pyarabic.araby as araby
import pandas as pd

def preprocess(text):
    cleaned_text = re.sub(r"http\S+", " ", text) # remove urls
    cleaned_text = re.sub(r"https\S+", " ", cleaned_text) # remove urls
    cleaned_text = re.sub(r"RT ", " ", cleaned_text) # remove rt
    cleaned_text= re.sub(r"@[\w]*", " ", cleaned_text) # remove handles
    cleaned_text=re.sub(r'[^0-9\u0600-\u06ff\u0750-\u077f\ufb50-\ufbc1\ufbd3-\ufd3f\ufd50-\ufd8f\ufd50-\ufd8f\ufe70-\ufefc\uFDF0-\uFDFD]+', ' ', cleaned_text)
    cleaned_text=araby.strip_diacritics(cleaned_text) 
    return cleaned_text  

def build_examples(args,line):
    fold_claims_df=pd.read_csv(args.fold_file,sep="\t")
    fold_claims= fold_claims_df["claim_id"].tolist()
    print(fold_claims)
    print(line['id'])
    if args.training and line["label"] == "NOT ENOUGH INFO":
        print(line['id'])
        return []
    
    if args.training:
      if line['id'] in fold_claims:
       print( line['id'] )
       pos_examples = []
       pos_examples_ids=[]
       for ev in line['evidence']:
         print(line['label'])
         if line['label']=='REFUTES':
          pos_examples.append([line['id'],preprocess(line['claim']),ev[0],ev[1],preprocess(ev[2]),1])
          pos_examples_ids.append(ev[1])
         elif line['label']=='SUPPORTS':
          pos_examples.append([line['id'],preprocess(line['claim']),ev[0],ev[1],preprocess(ev[2]),2])
          pos_examples_ids.append(ev[1])
         
       neg_examples=[]
       neg_num=len(pos_examples)*args.neg_ratio
       neg_timeline=[]
       for t in line['timeline']:
           if t[2] in pos_examples_ids:
              continue
           else:
              neg_timeline.append([line['id'],preprocess(line['claim']),t[0],t[1],preprocess(t[2]),0]) 
       if neg_num<=len(neg_timeline):
          neg_examples=random.sample(neg_timeline, neg_num)
       else:
          print("Number of negative examples in the timeline is less than negative ratio: ",len(neg_timeline))
          neg_examples=neg_timeline
       #print(len(neg_examples))
       all_examples=[]
       all_examples.extend(pos_examples)
       all_examples.extend(neg_examples)
       print("Number of examples is ",len(all_examples) )
       return all_examples
      else:
        return []
    else:
      all_examples=[]
      if line['id'] in fold_claims:
       for t in line['timeline']:
          all_examples.append([line['id'],preprocess(line['claim']),t[0],t[1],preprocess(t[2])])
       print("Number of examples is ",len(all_examples) )
       return all_examples
      else:
       return []        

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--neg_ratio", type=int, default=2)
    parser.add_argument("--fold_file",type=str,required=True)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--seed", type=int, default=3435)
    return parser.parse_args()


def main():
    args = build_args()
    random.seed(args.seed)
    lines = [line for line in jsonlines.open(args.in_file)]
    print(len(lines))
    out_examples = []
    for line in tqdm(lines, total=len(lines), desc="Building examples"):
        out_examples.extend(build_examples(args, line))

    print(f"Save to {args.out_file}")
    with io.open(args.out_file, "w", encoding="utf-8", errors="ignore") as out:
        for ex in out_examples:
            ex = list(map(str, ex))
            out.write("\t".join(ex) + "\n")


if __name__ == "__main__":
    main()
