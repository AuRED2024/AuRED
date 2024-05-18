### To run the BM25 model you need to follow the below steps:
1. Preprocess the data by running the following:
   > python prepare_data.py infile data/AuRED.json </br>
2. Index the processed data by running [index_data.sh](https://github.com/AuRED2024/AuRED/blob/main/code/BM25/index_data.sh)
   > bash index_data.sh </br>
3. Retrieve evidence for each rumor from its corresponding index by running [retrieve.sh](https://github.com/AuRED2024/AuRED/blob/main/code/BM25/retrieve.sh)
   > bash retrieve.sh </br>
4. Evaluate the output by running [evaluate.sh](https://github.com/AuRED2024/AuRED/blob/main/code/BM25/evaluate.sh)
   > bash evaluate.sh </br>
