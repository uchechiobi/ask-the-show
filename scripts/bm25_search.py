"""
BM25 keyword search - a second retrieval method alongside the Chroma vector
search from step 3. BM25 finds titles by matching the actual words in your
query (the classic search-engine approach), weighted so rare/distinctive
words count more than common ones. It's the retrieval method that reliably
catches an exact title like "Breaking Bad", which meaning-based vector
search can sometimes miss by drifting toward "similar theme" instead of
"exact match".

This is a shared module - other scripts import it, it isn't run directly.
"""

import json
import os
import re

from rank_bm25 import BM25Okapi

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "shows_with_plots.json")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def load_records():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(records):
    """Build a BM25 index with one document per title (its overview + plot)."""
    corpus_texts = [
        " ".join(part for part in [r.get("overview", ""), r.get("plot", "")] if part)
        for r in records
    ]
    tokenized_corpus = [tokenize(text) for text in corpus_texts]
    return BM25Okapi(tokenized_corpus)


def search(index, records, query, top_n=50):
    """Return the top_n titles by BM25 score for the query, as a list of
    (record, score) tuples, highest score first. Titles with a score of 0
    (no query words matched at all) are excluded."""
    scores = index.get_scores(tokenize(query))
    scored = [(record, score) for record, score in zip(records, scores) if score > 0]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_n]
