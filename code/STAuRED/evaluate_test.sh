data_dir="/data"
fold=1
lr=2e-05
 python "/code/evaluation/pyterrier_scorer.py" \
     --actual "${data_dir}/qrels/AuRED/test_fold${fold}_qrels_verifiable.txt" \
     --predicted "${out_dir}/MARBERTv2_AuRED_stance_fold${fold}_${lr}.txt" \
     --output_file "CV_results/MARBERTv2_AuRED_stance_test_results.csv"


python "/code/evaluation/FEVER_scorer.py" --predicted "${out_dir}/MARBERTv2_AuRED_stance_fold${fold}_${lr}.txt" \
       --actual "${data_dir}/qrels/AuRED/test_fold${fold}_qrels_verifiable.txt"

fold=2
lr=2e-05
 python "/code/evaluation/pyterrier_scorer.py" \
     --actual "${data_dir}/qrels/AuRED/test_fold${fold}_qrels_verifiable.txt" \
     --predicted "${out_dir}/MARBERTv2_AuRED_stance_fold${fold}_${lr}.txt" \
     --output_file "CV_results/MARBERTv2_AuRED_stance_test_results.csv"


python "/code/evaluation/FEVER_scorer.py" --predicted "${out_dir}/MARBERTv2_AuRED_stance_fold${fold}_${lr}.txt" \
       --actual "${data_dir}/qrels/AuRED/test_fold${fold}_qrels_verifiable.txt"

fold=3
lr=2e-05
 python "/code/evaluation/pyterrier_scorer.py" \
     --actual "${data_dir}/qrels/AuRED/test_fold${fold}_qrels_verifiable.txt" \
     --predicted "${out_dir}/MARBERTv2_AuRED_stance_fold${fold}_${lr}.txt" \
     --output_file "CV_results/MARBERTv2_AuRED_stance_test_results.csv"


python "/code/evaluation/FEVER_scorer.py" --predicted "${out_dir}/MARBERTv2_AuRED_stance_fold${fold}_${lr}.txt" \
       --actual "${data_dir}/qrels/AuRED/test_fold${fold}_qrels_verifiable.txt"

fold=4
lr=3e-05
 python "/code/evaluation/pyterrier_scorer.py" \
     --actual "${data_dir}/qrels/AuRED/test_fold${fold}_qrels_verifiable.txt" \
     --predicted "${out_dir}/MARBERTv2_AuRED_stance_fold${fold}_${lr}.txt" \
     --output_file "CV_results/MARBERTv2_AuRED_stance_test_results.csv"


python "/code/evaluation/FEVER_scorer.py" --predicted "${out_dir}/MARBERTv2_AuRED_stance_fold${fold}_${lr}.txt" \
       --actual "${data_dir}/qrels/AuRED/test_fold${fold}_qrels_verifiable.txt"

fold=5
lr=2e-05
 python "/code/evaluation/pyterrier_scorer.py" \
     --actual "${data_dir}/qrels/AuRED/test_fold${fold}_qrels_verifiable.txt" \
     --predicted "${out_dir}/MARBERTv2_AuRED_stance_fold${fold}_${lr}.txt" \
     --output_file "CV_results/MARBERTv2_AuRED_stance_test_results.csv"


python "/code/evaluation/FEVER_scorer.py" --predicted "${out_dir}/MARBERTv2_AuRED_stance_fold${fold}_${lr}.txt" \
       --actual "${data_dir}/qrels/AuRED/test_fold${fold}_qrels_verifiable.txt"
