import argparse
import json
import os
import time
import numpy as np
from tqdm import tqdm
import torch
import transformers
from sentence_transformers import CrossEncoder
import unicodedata
from urllib.parse import unquote
from cleantext import clean
import re
pt = re.compile(r"\[\[.*?\|(.*?)]]")
transformers.logging.set_verbosity_error()

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def clean_text(text):
    # text = re.sub(pt, r"\1", text)
    text = unquote(text)
    text = unicodedata.normalize('NFD', text)
    text = clean(text.strip(),
    fix_unicode=True,               # fix various unicode errors
    to_ascii=False,                 # transliterate to closest ASCII representation
    lower=False,                    # lowercase text
    no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
    no_urls=True,                   # replace all URLs with a special token
    no_emails=False,                # replace all email addresses with a special token
    no_phone_numbers=False,         # replace all phone numbers with a special token
    no_numbers=False,               # replace all numbers with a special token
    no_digits=False,                # replace all digits with a special token
    no_currency_symbols=False,      # replace all currency symbols with a special token
    no_punct=False,                 # remove punctuations
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_phone_number="<PHONE>",
    replace_with_number="<NUMBER>",
    replace_with_digit="0",
    replace_with_currency_symbol="<CUR>",
    lang="en"                       # set to 'de' for German special handling
    )
    return text

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

def retrieve_top_k_sentences(query, document, urls, bert_path, top_k, batch_size):
    # Load pre-trained RoBERTa model and tokenizer
    # ckpt = "bert_weights/nlp_corom_passage-ranking_english-base"
    # ckpt = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ckpt = bert_path
    model = CrossEncoder(ckpt, max_length=510, default_activation_function=torch.nn.Sigmoid())

    scores = []
    for i in tqdm(range(0, len(document), batch_size)):
        batch_sent = document[i:i+batch_size]
        batch = [ (query,s) for s in batch_sent ]

        scr = model.predict(batch)

        # Store scores
        scores.extend(scr)
        del scr
        torch.cuda.empty_cache()

    # Get the index of ranked sentences
    top_k_idx = np.argsort(scores)[::-1][:top_k]

    return [document[i] for i in top_k_idx], [urls[i] for i in top_k_idx], [scores[i] for i in top_k_idx]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Get top 100 sentences with BM25 in the knowledge store."
    )
    parser.add_argument(
        "-m",
        "--bert_path",
        type=str,
        default="",
        help="The path of the pretrained BERT model.",
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
    args.json_output = f"{args.json_output}_{args.start}_{args.end}.json"
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

                # Clean sentences with cleantext.clean_text()
                # It didnt improve the Sentence Retr HU-METEOR Score, so commented!!
                # document_in_sentences = [clean_text(s) for s in document_in_sentences]

                # Retrieve top_k sentences with tfidf
                st = time.time()
                top_k_sentences, top_k_urls, top_k_scores = retrieve_top_k_sentences(
                    example["claim"], document_in_sentences, sentence_urls, args.bert_path, args.top_k, args.batch_size
                )
                print(f"Top {args.top_k} retrieved. Time elapsed: {time.time() - st}.")

                json_data = {
                    "claim_id": idx,
                    "claim": example["claim"],
                    f"top_{args.top_k}": [
                        {"sentence": sent, "url": url, "score": str(score)}
                        for sent, url, score in zip(top_k_sentences, top_k_urls, top_k_scores)
                    ],
                }
                output_json.write(json.dumps(json_data, ensure_ascii=False) + "\n")
                done += 1
                output_json.flush()
