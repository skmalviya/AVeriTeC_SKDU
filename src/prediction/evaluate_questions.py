import argparse
import json
import scipy
import numpy as np
import sklearn
import nltk
from nltk import word_tokenize
from prediction.evaluate_veracity import AVeriTeCEvaluator, print_with_space

def eval_questions(args, in_file, split, top_k):
    print("Loading gold and pred questions...")
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
    print(f"Evaluating sentence retrieval on Answer-only score metric=(HU-{scorer.metric})...")
    for level in [5, 10, 50, 100, 150, 200]:
        if level <= top_k:
            score = scorer.evaluate_src_tgt(predictions, references, max_sent=level)
            print(f"Answer-only score metric=(HU-{scorer.metric}) level={level} : {score}")
            valid_scores.append(score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--top_k', type=int)

    args = parser.parse_args()


    # Extract pred sents
    predictions = []
    with open(args.input_path, "r", encoding="utf-8") as f_in:
        for idx,line in enumerate(f_in):
            js = json.loads(line)
            js_pred = {"claim_id": js["claim_id"], "claim": js["claim"]}
            evd = []
            for e in js["bm25_qau"]:
                evd.append({"question": e[0], "answer": e[1], "url": e[2]})
            js_pred["evidence"] = evd
            predictions.append(js_pred)
    # Extract pred sents

    with open(args.data_dir+'/'+args.split+'.json') as f:
        references = json.load(f)

    scorer = AVeriTeCEvaluator()
    q_score = scorer.evaluate_questions_only(predictions, references)
    print_with_space("Question-only score (HU-" + scorer.metric + "):", str(q_score))

    a_score = scorer.evaluate_answers_only(predictions, references)
    print_with_space("Answer-only score (HU-" + scorer.metric + "):", str(a_score))

    p_score = scorer.evaluate_questions_and_answers(predictions, references)
    print_with_space("Question-answer score (HU-" + scorer.metric + "):", str(p_score))
    print("====================")

    # eval_questions(args, args.input_path, args.split, args.top_k)