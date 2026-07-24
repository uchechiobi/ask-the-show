"""
Day 3: the four callable tools the agent can use.

Each function takes plain structured arguments and returns plain
structured data (dicts/lists) - none of them generate free-text prose.
That's what keeps the whole system "explainable": every fact the agent
reports back can be traced to one of these return values, not to
something a language model made up.

This is a shared module - other scripts import it, it isn't run directly.
"""

import bm25_search
import ranking
from data_access import get_chroma_collection

# Loaded once per process and reused, since building the BM25 index and
# loading the embedding model both take a moment.
_records_cache = None
_records_by_uid_cache = None
_bm25_index_cache = None
_collection_cache = None


def _records():
    global _records_cache, _records_by_uid_cache
    if _records_cache is None:
        _records_cache = bm25_search.load_records()
        _records_by_uid_cache = {r["uid"]: r for r in _records_cache}
    return _records_cache


def _records_by_uid():
    _records()  # ensure populated
    return _records_by_uid_cache


def _bm25_index():
    global _bm25_index_cache
    if _bm25_index_cache is None:
        _bm25_index_cache = bm25_search.build_index(_records())
    return _bm25_index_cache


def _collection():
    global _collection_cache
    if _collection_cache is None:
        _collection_cache = get_chroma_collection()
    return _collection_cache


def known_genres():
    """All distinct genre names present in the dataset."""
    return sorted({g for r in _records() for g in r.get("genres", [])})


def _brief(record):
    """A compact, consistent summary of a title - what tools return
    instead of the full raw record."""
    return {
        "uid": record["uid"],
        "title": record["title"],
        "year": record["year"],
        "media_type": record["media_type"],
        "genres": record.get("genres", []),
        "rating": record.get("rating"),
        "runtime": record.get("runtime"),
        "overview": record.get("overview", ""),
    }


# --- Tool 1: search_titles -------------------------------------------------

def search_titles(query, top_n=5):
    """
    Basic lookup by title or keyword.

    First checks whether any title in the catalog appears inside the
    query text (so "what is Severance about" still finds "Severance"),
    or the query is itself a fragment of a title. If nothing matches by
    title, falls back to BM25 keyword search over the overview/plot text.
    """
    records = _records()
    query_lower = query.lower().strip()

    title_in_query = [r for r in records if len(r["title"]) >= 3 and r["title"].lower() in query_lower]
    query_in_title = [r for r in records if query_lower and query_lower in r["title"].lower()]
    title_matches = title_in_query or query_in_title
    title_matches.sort(key=lambda r: len(r["title"]), reverse=bool(title_in_query))

    if title_matches:
        return {
            "method": "title_match",
            "results": [_brief(r) for r in title_matches[:top_n]],
        }

    hits = bm25_search.search(_bm25_index(), records, query, top_n=top_n)
    return {
        "method": "keyword_search",
        "results": [_brief(r) for r, _score in hits],
    }


# --- Tool 2: filter_by_constraints -----------------------------------------

def filter_by_constraints(genres=None, media_type=None, year_min=None, year_max=None,
                           max_runtime=None, min_rating=None, top_n=20):
    """
    Filter the whole catalog by structured constraints only - no query,
    no ranking. Every argument is optional; only the ones you pass are
    applied. Results are sorted by rating (highest first) since there's
    no relevance signal to sort by otherwise.

    Note: the dataset doesn't include total episode count (TMDB's
    "popular" list endpoint doesn't return it), so max_runtime (minutes)
    is the closest duration constraint available.
    """
    records = _records()
    genres_lower = {g.lower() for g in genres} if genres else None

    matches = []
    for r in records:
        if genres_lower and not (genres_lower & {g.lower() for g in r.get("genres", [])}):
            continue
        if media_type and r.get("media_type") != media_type:
            continue
        if year_min is not None and (r.get("year") is None or r["year"] < year_min):
            continue
        if year_max is not None and (r.get("year") is None or r["year"] > year_max):
            continue
        if max_runtime is not None and r.get("runtime") and r["runtime"] > max_runtime:
            continue
        if min_rating is not None and (r.get("rating") is None or r["rating"] < min_rating):
            continue
        matches.append(r)

    matches.sort(key=lambda r: r.get("rating") or 0, reverse=True)
    return {
        "count": len(matches),
        "results": [_brief(r) for r in matches[:top_n]],
    }


# --- Tool 3: compare_titles -------------------------------------------------

def compare_titles(title_a, title_b):
    """
    Look up two titles and return a structured, side-by-side comparison:
    rating gap, runtime gap, release-year gap, shared vs. distinct
    genres, and each title's own plot text verbatim - so differences in
    theme/tone are there for you to read, not summarized or guessed at.
    """
    result_a = search_titles(title_a, top_n=1)
    result_b = search_titles(title_b, top_n=1)

    if not result_a["results"]:
        return {"error": f"Couldn't find a title matching '{title_a}'"}
    if not result_b["results"]:
        return {"error": f"Couldn't find a title matching '{title_b}'"}

    record_a = _records_by_uid()[result_a["results"][0]["uid"]]
    record_b = _records_by_uid()[result_b["results"][0]["uid"]]

    genres_a = set(record_a.get("genres", []))
    genres_b = set(record_b.get("genres", []))

    runtime_diff = None
    if record_a.get("runtime") and record_b.get("runtime"):
        runtime_diff = record_a["runtime"] - record_b["runtime"]

    year_diff = None
    if record_a.get("year") and record_b.get("year"):
        year_diff = record_a["year"] - record_b["year"]

    return {
        "title_a": {**_brief(record_a), "plot": record_a.get("plot", "")},
        "title_b": {**_brief(record_b), "plot": record_b.get("plot", "")},
        "shared_genres": sorted(genres_a & genres_b),
        "only_in_a": sorted(genres_a - genres_b),
        "only_in_b": sorted(genres_b - genres_a),
        "rating_diff": round((record_a.get("rating") or 0) - (record_b.get("rating") or 0), 2),
        "runtime_diff_minutes": runtime_diff,
        "year_diff": year_diff,
    }


# --- Tool 4: recommend -------------------------------------------------

def recommend(query, genres=None, max_runtime=None, min_rating=None, top_n=10):
    """
    Day 2's hybrid retrieval (semantic + BM25) and weighted ranking
    formula, wrapped as a reusable function. Returns ranked suggestions
    for a free-text query, optionally narrowed by preferred genres, a
    max runtime, and/or a minimum rating.
    """
    records = _records()
    records_by_uid = _records_by_uid()

    results = _collection().query(query_texts=[query], n_results=50)
    sem_scores = {}
    for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
        similarity = max(0.0, 1.0 - distance)
        uid = meta["uid"]
        if uid not in sem_scores or similarity > sem_scores[uid]:
            sem_scores[uid] = similarity

    bm25_hits = bm25_search.search(_bm25_index(), records, query, top_n=50)
    bm25_scores = {r["uid"]: score for r, score in bm25_hits}

    candidate_uids = set(sem_scores) | set(bm25_scores)
    candidates = []
    for uid in candidate_uids:
        record = records_by_uid.get(uid)
        if record is None:
            continue
        if max_runtime is not None and record.get("runtime") and record["runtime"] > max_runtime:
            continue
        if min_rating is not None and (record.get("rating") is None or record["rating"] < min_rating):
            continue
        candidates.append({
            "record": record,
            "semantic_similarity": sem_scores.get(uid),
            "bm25_raw": bm25_scores.get(uid),
        })

    if not candidates:
        return {"results": []}

    scored = ranking.score_candidates(candidates, genres or [])
    results_out = []
    for s in scored[:top_n]:
        results_out.append({
            **_brief(s["record"]),
            "final_score": round(s["final_score"], 3),
            "semantic_similarity": round(s["semantic_similarity"], 3),
            "bm25_score": round(s["bm25_score"], 3),
            "text_relevance": round(s["text_relevance"], 3),
            "genre_match": round(s["genre_match"], 3),
            "rating_score": round(s["rating_score"], 3),
            "novelty_score": round(s["novelty_score"], 3),
        })
    return {"results": results_out}
