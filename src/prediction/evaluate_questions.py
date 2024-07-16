import argparse
import json
import scipy
import numpy as np
import sklearn
import nltk
from nltk import word_tokenize
from prediction.evaluate_veracity import AVeriTeCEvaluator, print_with_space


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--n_samples', type=int)
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
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
    for level in [3, 5, 10, 50, 100]:
        print("====================")
        scorer.max_questions = level
        print(f"For Level={level}...")
        q_score = scorer.evaluate_questions_only(predictions[args.start:args.end], references[args.start:args.end])
        print_with_space("Question-only score (HU-" + scorer.metric + "):", str(q_score))

        a_score = scorer.evaluate_answers_only(predictions[args.start:args.end], references[args.start:args.end])
        print_with_space("Answer-only score (HU-" + scorer.metric + "):", str(a_score))

        qa_score = scorer.evaluate_questions_and_answers(predictions[args.start:args.end], references[args.start:args.end])
        print_with_space("Question-answer score (HU-" + scorer.metric + "):", str(qa_score))
        print("====================")

