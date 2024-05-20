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
    data_tfidf = load_jsonl_data(args.in_tfidf)
    data_bm25 = load_jsonl_data(args.in_bm25)

    predicted_pages = []
    merge_evi_num = []
    page_num = 5

    with open(args.out_path, "w", encoding="utf-8") as f_out:
        for js1, js2 in tqdm(zip(data_tfidf, data_bm25)):
            assert js1["claim_id"] == js2["claim_id"]

            out_dict = {"claim_id": js1["claim_id"], "claim": js1["claim"], "top_200": js1["top_100"] + js2["top_100"]}
            f_out.write(json.dumps(out_dict, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--in_tfidf', type=str)
    parser.add_argument('--in_bm25', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--split', type=str, default="dev")
    parser.add_argument('--count', type=int, default=200)
    args = parser.parse_args()

    main(args)
    eval_sentences(args, args.out_path, args.split, args.count)