#Standard IR measures [P@5,R@5,MAP,F@5]
python pyterrier_scorer.py --predicted BM25_Top5.txt \
       --actual /data/qrels/AuRED/AuRED_qrels_verifiable.txt \
       --output_file BM25_scores.csv

#FEVER evaluation measures
python FEVER_scorer.py --predicted BM25_Top5.txt \
  --actual /data/qrels/AuRED/AuRED_qrels.txt


