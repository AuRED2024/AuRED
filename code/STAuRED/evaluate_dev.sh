for fold in 1 2 3 4 5  
do 
  for lr in 2e-05 3e-05 4e-05 5e-05
    do
     out_dir="./CV_runs/MARBERTv2_AuRED_stance"
     data_dir="/data/"

     python "/code/evaluation/pyterrier_scorer.py" \
     --actual "${data_dir}/qrels/AuRED/dev_fold${fold}_qrels_verifiable.txt" \
     --predicted "${out_dir}/MARBERTv2_AuRED_stance_fold${fold}_${lr}.txt" \
     --output_file "CV_results/MARBERTv2_AuRED_stance_finetuning_results.csv"

    done
done
