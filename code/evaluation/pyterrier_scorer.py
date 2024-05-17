import os
os.environ["JAVA_HOME"] = "PATH/jdk-11.0.7"
import pyterrier as pt
from pyterrier.measures import *
pt.init()
import glob
import pandas as pd
import argparse
from csv import writer

def evaluate_run(pred_path,golden_path):
     golden = pt.io.read_qrels(golden_path)
     pred=pt.io._read_results_trec(pred_path)
     eval=pt.Utils.evaluate(pred, golden , metrics = [P@5,R@5,MAP],perquery=False)
     return eval
     

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("--actual", "-g", required=True, type=str,
                        help="Path to file with gold annotations.")
   parser.add_argument("--predicted", "-p", required=True, type=str,
                        help="Path to file with predictions.")
   parser.add_argument("--output_file",required=False, type=str,
                        help="Path to csv results file.")

   args = parser.parse_args()
   gold_file = args.actual
   pred_file = args.predicted
   eval=evaluate_run(pred_file,gold_file)
   df_eval=pd.DataFrame([eval])
   print(df_eval)
   F1_score=2 * (df_eval['P@5'][0] * df_eval['R@5'][0]) / (df_eval['P@5'][0] + df_eval['R@5'][0])
   print("F1@5",F1_score)
   result_list=[args.predicted.split("/")[-1],df_eval['P@5'][0],df_eval['R@5'][0],df_eval['AP'][0],F1_score]
   with open(args.output_file, 'a') as f_object:
    writer_object = writer(f_object,delimiter ="\t")
    writer_object.writerow(result_list)
    f_object.close()
  
