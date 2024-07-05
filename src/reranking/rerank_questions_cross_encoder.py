import argparse
import json
import numpy as np
import torch
import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import CrossEncoder
from src.models.DualEncoderModule import DualEncoderModule


def triple_to_string(x):
    return " </s> ".join([item.strip() for item in x])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rerank the QA paris and keep top 3 QA paris as evidence using a pre-trained BERT model."
    )
    parser.add_argument(
        "-i",
        "--top_k_qa_file",
        default="data_store/dev_top_k_qa.json",
        help="Json file with claim and top k generated question-answer pairs.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        default="data_store/dev_top_3_rerank_qa.json",
        help="Json file with the top3 reranked questions.",
    )
    parser.add_argument(
        "-ckpt",
        "--best_checkpoint",
        type=str,
        default="pretrained_models/bert_dual_encoder.ckpt",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=3,
        help="top_n question answer pairs as evidence to keep.",
    )
    args = parser.parse_args()

    examples = []
    with open(args.top_k_qa_file) as f:
        for line in f:
            examples.append(json.loads(line))

    bert_model_name = "bert_weights/ms-marco-MiniLM-L-6-v2"

    model = CrossEncoder(bert_model_name, max_length=510, default_activation_function=torch.nn.Sigmoid())

    with open(args.output_file, "w", encoding="utf-8") as output_file:
        for example in tqdm.tqdm(examples):
            strs_to_score = []
            values = []

            bm25_qau = example["bm25_qau"] if "bm25_qau" in example else []
            claim = example["claim"]

            for question, answer, url in bm25_qau:
                str_to_score = (claim, question + answer)

                strs_to_score.append(str_to_score)
                values.append([question, answer, url])

            if len(bm25_qau) > 0:

                scores = model.predict(strs_to_score)

                # top_n = torch.argsort(scores, descending=True)[: args.top_n]
                top_n = np.argsort(scores)[::-1][:args.top_n]
                evidence = [
                    {
                        "question": values[i][0],
                        "answer": values[i][1],
                        "url": values[i][2],
                    }
                    for i in top_n
                ]
            else:
                evidence = []

            json_data = {
                "claim_id": example["claim_id"],
                "claim": claim,
                "evidence": evidence,
            }
            output_file.write(json.dumps(json_data, ensure_ascii=False) + "\n")
            output_file.flush()
