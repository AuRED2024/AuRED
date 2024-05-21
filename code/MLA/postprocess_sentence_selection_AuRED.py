# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import bisect
import jsonlines
from collections import defaultdict
from transformers.data.processors.utils import DataProcessor
import pandas as pd

def get_best_evidence(in_file, pred_sent_file, min_score):
    print(in_file, pred_sent_file)
    lines_0 = list(line for line in DataProcessor._read_tsv(in_file))
    lines_1 = list(
        float(line.strip()) for line in open(pred_sent_file, "r", encoding="utf-8-sig")
    )
    assert len(lines_0) == len(lines_1)

    best_evidence = defaultdict(lambda: [])
    for (line_0, line_1) in zip(lines_0, lines_1):
        assert len(line_0) == 5
        claim_id, claim, doc_id, sent_id, sent_text = line_0
        score = line_1
        claim_id, sent_id, score = claim_id, int(sent_id), float(score)
        if score > min_score:
            bisect.insort(best_evidence[claim_id], (-score, doc_id, sent_id,sent_text))

    for claim_id in best_evidence:
        for i, (score, doc_id, sent_id,sent_text) in enumerate(best_evidence[claim_id]):
            best_evidence[claim_id][i] = (doc_id, sent_id, -score,sent_text)

    return best_evidence


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--pred_sent_file", type=str, required=True)
    parser.add_argument("--pred_doc_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--max_evidence_per_claim", type=int, default=5)
    parser.add_argument("--min_score", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = build_args()
    best_evidence = get_best_evidence(args.in_file, args.pred_sent_file, args.min_score)
    print(best_evidence)
    print(f"Save to {args.out_file}")
    output_df=pd.DataFrame()
    claim_id=[]
    evidence_id=[]
    rank=[]
    score=[]
    for key in best_evidence:

        ranked_evidence=best_evidence[key][: args.max_evidence_per_claim]
        counter=1
        for ev in ranked_evidence:
            ev=list(ev)
            claim_id.append(key)
            evidence_id.append(ev[1])
            score.append(ev[2])
            rank.append(counter)
            counter=counter+1

    output_df['claim_id']=claim_id
    output_df['Q0']=['Q0']*len(claim_id)
    output_df['evidence_id']=evidence_id
    output_df['rank']=rank
    output_df['score']=score
    output_df['tag']=['MLA']*len(claim_id)
    output_df.to_csv(args.out_file,sep="\t",header=False,index=False)
    test_claims=list(set(output_df['claim_id'].tolist()))
    with jsonlines.open(args.pred_doc_file) as fin, jsonlines.open(
        args.out_json, "w"
    ) as out:
        for line in fin:
            claim_id = line["id"]
            if claim_id in (test_claims):
             line["predicted_evidence"] = best_evidence[claim_id][
                : args.max_evidence_per_claim
             ]
             out.write(line)


if __name__ == "__main__":
    main()
