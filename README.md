---
license: apache-2.0
---

# AVeriTeC


Data, knowledge store and source code to reproduce the baseline experiments for the [AVeriTeC](https://arxiv.org/abs/2305.13117) dataset, which will be used for the 7th [FEVER](https://fever.ai/) workshop co-hosted at EMNLP 2024.


### Set up environment

```
conda create -n averitec python=3.11
conda activate averitec

pip install -r requirements.txt
python -m spacy download en_core_web_lg
python -m nltk.downloader punkt
python -m nltk.downloader wordnet
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Scrape text from the URLs obtained by searching queries with the Google API.

We provide up to 1000 URLs for each claim returned from a Google API search using different queries. This is a courtesy aimed at reducing the cost of using the Google Search API for participants of the shared task. The URL files can be found [here](https://huggingface.co/chenxwh/AVeriTeC/tree/main/data_store/urls).

You can use your own scraping tool to extract sentences from the URLs. Alternatively, we have included a scraping tool for this purpose, which can be executed as follows. The processed files are also provided and can be found [here](https://huggingface.co/chenxwh/AVeriTeC/tree/main/data_store/knowledge_store).

```
bash script/scraper.sh <split> <start_idx> <end_idx> 
# e.g., bash script/scraper.sh dev 0 500
```

### Rank the sentences in the knowledge store with BM25
See [bm25_sentences.py](https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/reranking/bm25_sentences.py) for more args
```
python -m src.reranking.bm25_sentences
```