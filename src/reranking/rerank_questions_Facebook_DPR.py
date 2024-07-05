import argparse
import json
import torch
import numpy as np
import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from src.models.DualEncoderModule import DualEncoderModule

from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
from sklearn.metrics.pairwise import cosine_similarity

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

    # Initialize the DPR question and context encoders
    question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

    # Initialize the tokenizers
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

    with open(args.output_file, "w", encoding="utf-8") as output_file:
        for example in tqdm.tqdm(examples):
            qa_pairs = []
            values = []

            bm25_qau = example["bm25_qau"] if "bm25_qau" in example else []
            claim = example["claim"]

            for question, answer, url in bm25_qau:
                qa_pair = {"question": question, "answer":answer}

                qa_pairs.append(qa_pair)
                values.append([question, answer, url])

            if len(bm25_qau) > 0:

                # Encode the claim
                claim_inputs = question_tokenizer(claim, return_tensors="pt")
                claim_embedding = question_encoder(**claim_inputs).pooler_output

                # Encode the question-answer pairs
                qa_embeddings = []
                for qa in qa_pairs:
                    context = f"Q: {qa['question']} A: {qa['answer']}"
                    context_inputs = context_tokenizer(context, return_tensors="pt")
                    context_embedding = context_encoder(**context_inputs).pooler_output
                    qa_embeddings.append(context_embedding)

                # Calculate cosine similarities
                qa_embeddings = torch.cat(qa_embeddings, dim=0)
                scores = cosine_similarity(claim_embedding.detach().numpy(),
                                                 qa_embeddings.detach().numpy()).flatten()

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
