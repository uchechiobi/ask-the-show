"""
Step 5: Interactive hybrid search + ranking test.

Enter a free-text query plus optional constraints (preferred genres, max
runtime), and see the top-ranked titles with every score component broken
down so you can see exactly how the ranking formula produced that order.

Run this after 02_fetch_wikipedia.py and 03_chunk_and_embed.py have built
data/shows_with_plots.json and the Chroma database.
"""

import os

import chromadb
from chromadb.utils import embedding_functions

import bm25_search
import ranking

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
COLLECTION_NAME = "ask_the_show"

SEMANTIC_TOP_N = 50   # how many chunks to pull from Chroma per query
BM25_TOP_N = 50        # how many titles to pull from BM25 per query
RESULTS_TO_SHOW = 10


def semantic_candidates(collection, query, top_n):
    """Query Chroma and collapse chunk-level hits down to one best
    (highest-similarity) result per title."""
    results = collection.query(query_texts=[query], n_results=top_n)
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    best_per_title = {}
    for meta, distance in zip(metadatas, distances):
        similarity = max(0.0, 1.0 - distance)  # cosine distance -> similarity
        uid = meta["uid"]
        if uid not in best_per_title or similarity > best_per_title[uid]:
            best_per_title[uid] = similarity
    return best_per_title


def parse_genres(raw):
    return [g.strip() for g in raw.split(",") if g.strip()] if raw.strip() else []


def main():
    print("Loading embedding model and Chroma collection...")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

    print("Loading BM25 keyword index...")
    records = bm25_search.load_records()
    records_by_id = {r["uid"]: r for r in records}
    bm25_index = bm25_search.build_index(records)

    all_genres = sorted({g for r in records for g in r.get("genres", [])})
    print(f"Ready. {len(records)} titles loaded.")
    print(f"Available genres: {', '.join(all_genres)}\n")

    while True:
        query = input("Query (or 'quit'): ").strip()
        if query.lower() in ("quit", "exit", ""):
            break

        requested_genres = parse_genres(input("  Preferred genres, comma-separated (optional): "))

        max_runtime_raw = input("  Max runtime in minutes (optional): ").strip()
        max_runtime = int(max_runtime_raw) if max_runtime_raw else None

        # --- Retrieval: gather candidates from both search methods ---
        sem_scores = semantic_candidates(collection, query, SEMANTIC_TOP_N)
        bm25_hits = bm25_search.search(bm25_index, records, query, top_n=BM25_TOP_N)
        bm25_scores = {r["uid"]: score for r, score in bm25_hits}

        candidate_uids = set(sem_scores) | set(bm25_scores)
        candidates = []
        for uid in candidate_uids:
            record = records_by_id.get(uid)
            if record is None:
                continue
            if max_runtime is not None and record.get("runtime") and record["runtime"] > max_runtime:
                continue  # hard filter
            candidates.append({
                "record": record,
                "semantic_similarity": sem_scores.get(uid),
                "bm25_raw": bm25_scores.get(uid),
            })

        if not candidates:
            print("  No candidates matched your constraints.\n")
            continue

        # --- Ranking: score and sort the candidate pool ---
        scored = ranking.score_candidates(candidates, requested_genres)

        print()
        for rank, s in enumerate(scored[:RESULTS_TO_SHOW], start=1):
            r = s["record"]
            print(f"{rank}. {r['title']} ({r['year']}) [{r['media_type']}] "
                  f"- final score {s['final_score']:.3f}")
            print(f"   genres: {', '.join(r.get('genres', [])) or 'n/a'} | "
                  f"rating: {r.get('rating')} | runtime: {r.get('runtime')} min")
            print(f"   components -> semantic: {s['semantic_similarity']:.3f}  "
                  f"bm25: {s['bm25_score']:.3f}  text_relevance: {s['text_relevance']:.3f}  "
                  f"genre_match: {s['genre_match']:.3f}  rating: {s['rating_score']:.3f}  "
                  f"novelty: {s['novelty_score']:.3f}")
            print()


if __name__ == "__main__":
    main()
