"""
The weighted ranking formula: combines four signals into one score per
title, so we can explain exactly why something was recommended.

    final_score = W_RELEVANCE * text_relevance
                + W_GENRE     * genre_match
                + W_RATING    * rating_score
                + W_NOVELTY   * novelty_score

text_relevance is itself a blend of two retrieval signals - this is the
"hybrid" part of hybrid retrieval:

    text_relevance = HYBRID_ALPHA       * semantic_similarity
                    + (1 - HYBRID_ALPHA) * bm25_score

Every component is normalized to 0-1 so the weights are directly
comparable. Tune the constants below to change how results are ranked.

This is a shared module - other scripts import it, it isn't run directly.
"""

# --- Tunable weights (sum to 1.0 so the final score stays in 0-1) ---
W_RELEVANCE = 0.50   # how well the text matches your query (semantic + keyword)
W_GENRE = 0.20        # how well the title's genres match what you asked for
W_RATING = 0.15       # TMDB audience rating (higher = more broadly liked)
W_NOVELTY = 0.15      # how recent the title is (higher = more novel/fresh)

# Within text_relevance, how much to trust meaning-based search vs. exact keywords.
HYBRID_ALPHA = 0.6    # 0.6 semantic, 0.4 keyword (BM25)


def normalize(values):
    """Min-max normalize a list of numbers to 0-1. If every value is equal,
    returns 0.5 for all of them (avoids a divide-by-zero)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def genre_match_score(record_genres, requested_genres):
    """Fraction of requested genres this title actually has. If no genres
    were requested, this component doesn't penalize anything (returns 1.0)."""
    if not requested_genres:
        return 1.0
    record_set = {g.lower() for g in record_genres}
    requested_set = {g.lower() for g in requested_genres}
    return len(record_set & requested_set) / len(requested_set)


def rating_score(rating):
    """TMDB rating is 0-10; normalize to 0-1."""
    if rating is None:
        return 0.0
    return max(0.0, min(1.0, rating / 10.0))


def novelty_score(year, min_year, max_year):
    """Newer titles score higher, relative to the years present in the
    current candidate pool. This is a simple proxy for 'novelty' - a
    more recent release is treated as fresher / less likely to be
    something you've already seen many times."""
    if year is None or max_year is None or max_year == min_year:
        return 0.5
    return (year - min_year) / (max_year - min_year)


def score_candidates(candidates, requested_genres):
    """
    candidates: list of dicts with keys:
        record (the title's data dict), semantic_similarity (0-1 or None),
        bm25_raw (float or None)

    Returns a new list of dicts (one per candidate) with every score
    component included, sorted by final_score descending.
    """
    bm25_raw_values = [c["bm25_raw"] or 0.0 for c in candidates]
    bm25_norm_values = normalize(bm25_raw_values)

    years = [c["record"]["year"] for c in candidates if c["record"].get("year")]
    min_year = min(years) if years else None
    max_year = max(years) if years else None

    scored = []
    for candidate, bm25_norm in zip(candidates, bm25_norm_values):
        record = candidate["record"]
        semantic_sim = candidate["semantic_similarity"] or 0.0

        text_relevance = HYBRID_ALPHA * semantic_sim + (1 - HYBRID_ALPHA) * bm25_norm
        genre = genre_match_score(record.get("genres", []), requested_genres)
        rating = rating_score(record.get("rating"))
        novelty = novelty_score(record.get("year"), min_year, max_year)

        final_score = (
            W_RELEVANCE * text_relevance
            + W_GENRE * genre
            + W_RATING * rating
            + W_NOVELTY * novelty
        )

        scored.append({
            "record": record,
            "semantic_similarity": semantic_sim,
            "bm25_score": bm25_norm,
            "text_relevance": text_relevance,
            "genre_match": genre,
            "rating_score": rating,
            "novelty_score": novelty,
            "final_score": final_score,
        })

    scored.sort(key=lambda s: s["final_score"], reverse=True)
    return scored
