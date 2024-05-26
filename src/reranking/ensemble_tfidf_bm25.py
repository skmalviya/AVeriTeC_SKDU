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
    sent_num = args.count

    with open(args.out_path, "w", encoding="utf-8") as f_out:
        for js1, js2 in tqdm(zip(data_tfidf, data_bm25)):
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

            # for js1, js2 in tqdm(zip(data1, data2)):
            #     cand_pages1 = [cand_id for cand_id, score in
            #                    remove_unmatch_year(js1["predicted_pages"][:5], js1["claim"])[:sent_num]]
            #     cand_pages2 = [cand_id for cand_id, score in
            #                    remove_unmatch_year(js2["predicted_pages"][:5], js1["claim"])[:sent_num]]
            #
            #     # rank sum
            #     score_dict = {}
            #     for rk, cand_id in enumerate(cand_pages1):
            #         score_dict[cand_id] = [rk]
            #     for rk, cand_id in enumerate(cand_pages2):
            #         if cand_id in score_dict:
            #             score_dict[cand_id].append(rk)
            #         else:
            #             score_dict[cand_id] = [rk]
            #
            #     cand_pages = []
            #     for cand_id in score_dict:
            #         if len(score_dict[cand_id]) == 1:
            #             score_dict[cand_id].append(sent_num)
            #         cand_pages.append((average(score_dict[cand_id]), cand_id))
            #     cand_pages = sorted(cand_pages)
            #     cand_pages = polish_cand_pages(cand_pages, js1["claim"], doc_db)
            #     merge_evi_num.append(len(cand_pages))
            #     res_pages = [[cp[1], cp[0]] for cp in cand_pages[:sent_num]]
            #
            #     oentry = {
            #         "id": js2["id"],
            #         "claim": js1["claim"],
            #         "predicted_pages": res_pages
            #     }
            #     predicted_pages.append(oentry)
            #
            # save_path = f"data/{args.split}.pages.hybrank.roberta.bm25.p5.jsonl"
            # print(f"save to {save_path}")
            # from my_utils import save_jsonl_data
            # save_jsonl_data(predicted_pages, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--in_tfidf', type=str)
    parser.add_argument('--in_bm25', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--split', type=str, default="dev")
    parser.add_argument('--count', type=int, default=100)
    args = parser.parse_args()

    main(args)
    eval_sentences(args, args.out_path, args.split, args.count)