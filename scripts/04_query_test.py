"""
Step 4: Interactive test — type a question, see the top 5 retrieved chunks.

Run this after 03_chunk_and_embed.py has built the Chroma database.
"""

import os

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
COLLECTION_NAME = "ask_the_show"
TOP_K = 5


def main():
    print("Loading local embedding model (all-MiniLM-L6-v2)...")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
    print(f"Collection loaded: {collection.count()} chunks available.\n")

    while True:
        question = input("Ask a question (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit", ""):
            break

        results = collection.query(query_texts=[question], n_results=TOP_K)

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        print()
        for rank, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
            snippet = doc[:300].replace("\n", " ")
            print(f"{rank}. {meta['title']} ({meta['year']}) "
                  f"[{meta['media_type']}] - distance {dist:.3f}")
            print(f"   {snippet}...")
            print()


if __name__ == "__main__":
    main()
