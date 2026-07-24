"""
Step 3: Chunk each title's text (~500 tokens per chunk) and embed it into a
local Chroma vector database.

Input:  data/shows_with_plots.json
Output: a Chroma database on disk at data/chroma_db/
"""

import json
import os

import chromadb
import tiktoken
from chromadb.utils import embedding_functions

INPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "shows_with_plots.json")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
COLLECTION_NAME = "ask_the_show"

CHUNK_SIZE = 500     # tokens per chunk
CHUNK_OVERLAP = 50   # tokens shared between consecutive chunks, for context
ADD_BATCH_SIZE = 100 # how many chunks to hand to Chroma at once

encoding = tiktoken.get_encoding("cl100k_base")


def chunk_text(text):
    """Split text into ~CHUNK_SIZE-token pieces with CHUNK_OVERLAP tokens
    of overlap between consecutive chunks."""
    tokens = encoding.encode(text)
    if not tokens:
        return []

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens))
        if end >= len(tokens):
            break
        start = end - CHUNK_OVERLAP

    return chunks


def build_chunk_records(records):
    """Turn each show/movie record into one or more (id, document, metadata) chunks."""
    ids, documents, metadatas = [], [], []

    for record in records:
        text = "\n\n".join(
            part for part in [record.get("overview", ""), record.get("plot", "")] if part
        ).strip()
        if not text:
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            ids.append(f"{record['uid']}_{i}")
            documents.append(chunk)
            metadatas.append({
                "uid": record["uid"],
                "tmdb_id": record["tmdb_id"],
                "title": record["title"],
                "year": record["year"] or 0,
                "media_type": record["media_type"],
                "genres": ", ".join(record.get("genres", [])),
                "rating": record.get("rating") or 0.0,
                "runtime": record.get("runtime") or 0,
                "chunk_index": i,
            })

    return ids, documents, metadatas


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Chunking text for {len(records)} titles...")
    ids, documents, metadatas = build_chunk_records(records)
    print(f"Created {len(ids)} chunks.")

    print("Loading local embedding model (all-MiniLM-L6-v2)...")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        # Cosine distance ranges 0 (identical) to 2 (opposite), so
        # similarity = 1 - distance is directly usable as a 0-1 score.
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Embedding and storing {len(ids)} chunks in batches of {ADD_BATCH_SIZE}...")
    for start in range(0, len(ids), ADD_BATCH_SIZE):
        end = start + ADD_BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  added {min(end, len(ids))}/{len(ids)} chunks")

    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} chunks, "
          f"stored at {CHROMA_PATH}")


if __name__ == "__main__":
    main()
