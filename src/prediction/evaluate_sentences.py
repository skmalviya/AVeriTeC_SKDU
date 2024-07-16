import argparse
import json
import scipy
import numpy as np
import sklearn
import nltk
from nltk import word_tokenize
from prediction.evaluate_veracity import AVeriTeCEvaluator

def eval_sentences(args, in_file, split, top_k=100):
    print("Loading gold and pred sentence...")
    # Extract gold sents
    references = []
    gold_file = '{0}/{1}.json'.format(args.data_dir,split)
    with open(gold_file, "r", encoding="utf-8") as f_gold:
        for gold_data in json.load(f_gold):
            references.append( [a["answer"] for qa in gold_data["questions"] \
                                for a in qa["answers"]] if split != 'test' else [] )

    # Extract pred sents
    predictions = []
    with open(in_file, "r", encoding="utf-8") as f_in:
        for idx,line in enumerate(f_in):
            js = json.loads(line)
            predictions.append([e["sentence"] for e in js["top_"+str(top_k)]])

    scorer = AVeriTeCEvaluator()
    valid_scores = []
    print("Total data to eval = ", len(references))
    print(f"Evaluating sentence retrieval on Answer-only score metric=(HU-{scorer.metric})...")
    for level in [3, 5, 10, 50, 100, 150, 200, 500, 750, 1000]:
        if level <= top_k:
            score = scorer.evaluate_src_tgt(predictions[args.start:args.end], references[args.start:args.end], max_sent=level)
            print(f"Answer-only score metric=(HU-{scorer.metric}) level={level} : {score}")
            valid_scores.append(score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--top_k', type=int)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)

    args = parser.parse_args()
    print("Loading sentences file...", args.input_path)
    eval_sentences(args, args.input_path, args.split, args.top_k)