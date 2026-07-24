"""
Step 1: Pull popular movies and TV shows from TMDB.

For each title we keep: title, overview, genre(s), year, rating, runtime.
Output: data/tmdb_shows.json
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

# How many titles to pull of each type. 750 + 750 = 1500, inside the 1,000-2,000 target.
# Each TMDB page holds 20 results, so we convert the target count into a page count.
NUM_MOVIES = 750
NUM_TV_SHOWS = 750
MOVIE_PAGES = -(-NUM_MOVIES // 20)  # ceiling division
TV_PAGES = -(-NUM_TV_SHOWS // 20)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tmdb_shows.json")

session = requests.Session()


def api_get(path, params=None):
    """GET from the TMDB API, with a couple of retries if we get rate-limited."""
    params = dict(params or {})
    params["api_key"] = API_KEY
    url = f"{BASE_URL}{path}"

    for attempt in range(5):
        response = session.get(url, params=params, timeout=15)
        if response.status_code == 429:
            # TMDB says how many seconds to wait in the Retry-After header.
            wait = int(response.headers.get("Retry-After", 2))
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()

    raise RuntimeError(f"Gave up on {url} after repeated rate-limit errors")


def fetch_genre_map(media_type):
    """Returns {genre_id: genre_name} for 'movie' or 'tv'."""
    data = api_get(f"/genre/{media_type}/list")
    return {g["id"]: g["name"] for g in data["genres"]}


def fetch_popular(media_type, num_pages):
    """Fetch `num_pages` pages of the popular list for 'movie' or 'tv'."""
    results = []
    for page in range(1, num_pages + 1):
        data = api_get(f"/{media_type}/popular", params={"page": page})
        results.extend(data["results"])
        print(f"  fetched {media_type} page {page}/{num_pages} "
              f"({len(results)} titles so far)")
    return results


def fetch_runtime(media_type, tmdb_id):
    """Fetch the details endpoint just to get runtime (not in the popular list)."""
    try:
        data = api_get(f"/{media_type}/{tmdb_id}")
    except requests.HTTPError:
        return None

    if media_type == "movie":
        return data.get("runtime")

    # TV shows report runtime per episode, as a list. Take the first value.
    episode_runtimes = data.get("episode_run_time") or []
    return episode_runtimes[0] if episode_runtimes else None


def build_records(raw_items, media_type, genre_map):
    """Turn raw TMDB list results into our simplified record format, with runtime
    filled in via parallel detail requests."""
    records = []
    for item in raw_items:
        title = item.get("title") if media_type == "movie" else item.get("name")
        date = item.get("release_date") if media_type == "movie" else item.get("first_air_date")
        year = int(date[:4]) if date else None
        genres = [genre_map.get(gid, "Unknown") for gid in item.get("genre_ids", [])]

        records.append({
            "tmdb_id": item["id"],
            "media_type": media_type,
            "title": title,
            "overview": item.get("overview", ""),
            "genres": genres,
            "year": year,
            "rating": item.get("vote_average"),
            "runtime": None,  # filled in below
        })

    print(f"  fetching runtime for {len(records)} {media_type} titles...")
    with ThreadPoolExecutor(max_workers=10) as pool:
        future_to_record = {
            pool.submit(fetch_runtime, media_type, r["tmdb_id"]): r
            for r in records
        }
        done = 0
        for future in as_completed(future_to_record):
            record = future_to_record[future]
            record["runtime"] = future.result()
            done += 1
            if done % 100 == 0:
                print(f"    runtime lookups: {done}/{len(records)}")

    return records


def main():
    if not API_KEY:
        raise SystemExit(
            "TMDB_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    print("Fetching genre lists...")
    movie_genres = fetch_genre_map("movie")
    tv_genres = fetch_genre_map("tv")

    print(f"Fetching {MOVIE_PAGES} pages of popular movies...")
    raw_movies = fetch_popular("movie", MOVIE_PAGES)

    print(f"Fetching {TV_PAGES} pages of popular TV shows...")
    raw_tv = fetch_popular("tv", TV_PAGES)

    movie_records = build_records(raw_movies, "movie", movie_genres)
    tv_records = build_records(raw_tv, "tv", tv_genres)

    all_records = movie_records + tv_records

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_records)} titles to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
