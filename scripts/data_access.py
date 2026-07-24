"""
Shared helper for loading the Chroma collection the same way every script
needs it (Day 1's query test, Day 2's ranked search, Day 3's tools all
use this). Not something you run directly.
"""

import os

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
COLLECTION_NAME = "ask_the_show"


def get_chroma_collection():
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
