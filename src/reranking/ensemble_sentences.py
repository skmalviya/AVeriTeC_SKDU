from my_utils import load_jsonl_data
import argparse
import os
import json
from tqdm import tqdm
import re
from my_utils import average
import nltk
from prediction.evaluate_sentences import eval_sentences

def main(args):

    # merge two results
    data_f1 = load_jsonl_data(args.in_f1)
    data_f2 = load_jsonl_data(args.in_f2)

    predicted_pages = []
    merge_evi_num = []
    sent_num = args.count

    with open(args.out_path, "w", encoding="utf-8") as f_out:
        for js1, js2 in tqdm(zip(data_f1, data_f2)):
            assert js1["claim_id"] == js2["claim_id"]

            cand_sents_dict = {}
            for s in js1["top_100"]+js2["top_100"]:
                if s["sentence"] not in cand_sents_dict:
                    cand_sents_dict[s["sentence"]] = s
                    
            cand_sents1 = [sent["sentence"] for sent in js1["top_100"][:sent_num]]
            cand_sents2 = [sent["sentence"] for sent in js2["top_100"][:sent_num]]

            # rank sum
            score_dict = {}
            for rk, cand_id in enumerate(cand_sents1):
                score_dict[cand_id] = [rk]
            for rk, cand_id in enumerate(cand_sents2):
                if cand_id in score_dict:
                    score_dict[cand_id].append(rk)
                else:
                    score_dict[cand_id] = [rk]

            cand_sents = []
            for cand_id in score_dict:
                if len(score_dict[cand_id]) == 1:
                    score_dict[cand_id].append(sent_num)
                cand_sents.append((average(score_dict[cand_id]), cand_id))
            cand_sents = sorted(cand_sents)
            res_sents = [ cand_sents_dict[cp[1]] for cp in cand_sents[:sent_num]]
            js1["top_100"] = res_sents

            f_out.write(json.dumps(js1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--in_f1', type=str)
    parser.add_argument('--in_f2', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--split', type=str, default="dev")
    parser.add_argument('--count', type=int, default=100)
    args = parser.parse_args()

    main(args)
    eval_sentences(args, args.out_path, args.split, args.count)