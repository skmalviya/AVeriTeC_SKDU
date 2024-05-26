import argparse
import json
import os
import time
import numpy as np
import nltk
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, RobertaModel
import torch


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def combine_all_sentences(knowledge_file):
    sentences, urls = [], []

    with open(knowledge_file, "r", encoding="utf-8") as json_file:
        for i, line in enumerate(json_file):
            data = json.loads(line)
            sentences.extend(data["url2text"])
            urls.extend([data["url"] for i in range(len(data["url2text"]))])
    return sentences, urls, i + 1


def get_sentence_embedding(model, tokenizer, sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the mean of the output embeddings
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def get_sentence_embeddings_in_batches(model, tokenizer, sentences, batch_size=32):
    model.to(device)  # Move model to GPU

    embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=510).to(device)  # Move inputs to GPU

        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean of the token embeddings
        embeddings.append(batch_embeddings)
        del batch_embeddings
        torch.cuda.empty_cache()
    return torch.cat(embeddings, dim=0).cpu()  # Concatenate and move all embeddings to CPU at once

def retrieve_top_k_sentences(query, document, urls, top_k, batch_size):
    # Load pre-trained RoBERTa model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert_weights/nlp_corom_sentence-embedding_english-base')
    model = RobertaModel.from_pretrained('bert_weights/nlp_corom_sentence-embedding_english-base')

    # Get the embedding for the claim
    claim_embedding = get_sentence_embedding(model, tokenizer, query).unsqueeze(0)

    # Get embeddings for all sentences in the pool
    # sentence_embeddings = torch.stack(
    #     [get_sentence_embedding(model, tokenizer, sentence) for sentence in tqdm(document)])

    # Get embeddings for all sentences in the pool
    sentence_embeddings = get_sentence_embeddings_in_batches(model, tokenizer, document, batch_size)

    # Compute the cosine similarity between the claim and each sentence
    scores = cosine_similarity(claim_embedding.numpy(), sentence_embeddings.numpy()).flatten()
    top_k_idx = np.argsort(scores)[::-1][:top_k]

    return [document[i] for i in top_k_idx], [urls[i] for i in top_k_idx]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Get top 100 sentences with BM25 in the knowledge store."
    )
    parser.add_argument(
        "-k",
        "--knowledge_store_dir",
        type=str,
        default="data_store/output_dev",
        help="The path of the knowledge_store_dir containing json files with all the retrieved sentences.",
    )
    parser.add_argument(
        "-c",
        "--claim_file",
        type=str,
        default="data/dev.json",
        help="The path of the file that stores the claim.",
    )
    parser.add_argument(
        "-o",
        "--json_output",
        type=str,
        default="data_store/dev_top_k.json",
        help="The output dir for JSON files to save the top 100 sentences for each claim.",
    )
    parser.add_argument(
        "--top_k",
        default=100,
        type=int,
        help="How many documents should we pick out with BM25.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="What should be the batch size for embedding generation.",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=0,
        help="Staring index of the files to process.",
    )
    parser.add_argument(
        "-e", "--end", type=int, default=-1, help="End index of the files to process."
    )

    args = parser.parse_args()

    with open(args.claim_file, "r", encoding="utf-8") as json_file:
        target_examples = json.load(json_file)

    if args.end == -1:
        args.end = len(os.listdir(args.knowledge_store_dir))
        print(args.end)

    files_to_process = list(range(args.start, args.end))
    total = len(files_to_process)

    with open(args.json_output, "w", encoding="utf-8") as output_json:
        done = 0
        for idx, example in enumerate(target_examples):
            # Load the knowledge store for this example
            if idx in files_to_process:
                print(f"Processing claim {idx}... Progress: {done + 1} / {total}")
                document_in_sentences, sentence_urls, num_urls_this_claim = (
                    combine_all_sentences(
                        os.path.join(args.knowledge_store_dir, f"{idx}.json")
                    )
                )

                print(
                    f"Obtained {len(document_in_sentences)} sentences from {num_urls_this_claim} urls."
                )

                # Retrieve top_k sentences with tfidf
                st = time.time()
                top_k_sentences, top_k_urls = retrieve_top_k_sentences(
                    example["claim"], document_in_sentences, sentence_urls, args.top_k, args.batch_size
                )
                print(f"Top {args.top_k} retrieved. Time elapsed: {time.time() - st}.")

                json_data = {
                    "claim_id": idx,
                    "claim": example["claim"],
                    f"top_{args.top_k}": [
                        {"sentence": sent, "url": url}
                        for sent, url in zip(top_k_sentences, top_k_urls)
                    ],
                }
                output_json.write(json.dumps(json_data, ensure_ascii=False) + "\n")
                done += 1
                output_json.flush()
