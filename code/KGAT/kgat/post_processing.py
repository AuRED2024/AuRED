import jsonlines
from json import load
import json 
import argparse
# Opening JSON file
 
# returns JSON object as 
# a dictionary
data_labels_path='/kgat/output/MARBERTv2/test_lr2e-5_5epochs_fold1.json'
data_evidence_path='/retrieval_model/output/CV_experiments/verification_data/test_lr2e-5_5epochs_fold1.json'
 
# Iterating through the json
# list
def write_data(data,filename):
   with open(filename, 'a') as f:
       json.dump(data, f,ensure_ascii = False)
       f.write('\n')

# labels_dict={}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--predicted_evidence')
    parser.add_argument('--predicted_labels')
    parser.add_argument('--output_file')

    args = parser.parse_args()
    with open(args.predicted_labels) as fin:
     for step, line in enumerate(fin):
            instance = json.loads(line.strip())
            with open(args.predicted_evidence) as fin:
              for step, line in enumerate(fin):
                 ev_instance = json.loads(line.strip())
                 if ev_instance['id']== instance['id'] :
                    instance.update(ev_instance)
                    instance['predicted_evidence'] = instance.pop('evidence')
                    write_data(instance,args.output_file)
