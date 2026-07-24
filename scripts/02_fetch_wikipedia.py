"""
Step 2: Look up a Wikipedia plot summary for each title from step 1.

Input:  data/tmdb_shows.json
Output: data/shows_with_plots.json (same records, plus a "plot" field)

Safe to interrupt and re-run: already-processed titles are skipped.
"""

import json
import os
import re
import time

import wikipedia

# Wikipedia's API rejects requests without a descriptive User-Agent (returns a
# 403 robot-policy error instead of JSON). Identify ourselves so requests go through.
wikipedia.set_user_agent(
    "AskTheShow/1.0 (educational project; https://github.com/uchechiobi/ask-the-show)"
)

INPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tmdb_shows.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "shows_with_plots.json")

SAVE_EVERY = 50
PLOT_HEADINGS = ("plot", "plot summary", "synopsis", "premise")


def extract_plot_section(page_content):
    """Wikipedia page content uses '== Heading ==' markers. Pull out the text
    between a Plot-like heading and the next heading of the same level."""
    lines = page_content.split("\n")
    plot_lines = []
    in_plot = False

    for line in lines:
        heading_match = re.match(r"^(==+)\s*(.+?)\s*\1$", line.strip())
        if heading_match:
            heading_text = heading_match.group(2).strip().lower()
            if heading_text in PLOT_HEADINGS:
                in_plot = True
                continue
            elif in_plot:
                break  # hit the next heading, so the plot section is over
        elif in_plot:
            plot_lines.append(line)

    return "\n".join(plot_lines).strip()


def _load_page(title_candidate):
    """Fetch a Wikipedia page by exact title. Retries once on a transient
    network/API hiccup before giving up on this candidate."""
    for attempt in range(2):
        try:
            return wikipedia.page(title_candidate, auto_suggest=False)
        except wikipedia.exceptions.DisambiguationError as e:
            if not e.options:
                return None
            try:
                return wikipedia.page(e.options[0], auto_suggest=False)
            except Exception:
                return None
        except (wikipedia.exceptions.PageError, wikipedia.exceptions.WikipediaException):
            return None
        except Exception:
            # Likely a transient network/JSON error from Wikipedia's API - retry once.
            time.sleep(1)
    return None


def _page_text(page):
    """Get the best available text for a page: its Plot section if present,
    otherwise its intro summary. Returns None if both requests fail."""
    try:
        plot = extract_plot_section(page.content)
    except Exception:
        plot = None

    if plot:
        return plot

    try:
        return page.summary
    except Exception:
        return None


def find_plot(title, year, media_type):
    """Try exact disambiguated titles first, then fall back to Wikipedia's
    own search index for the closest matching page."""
    candidates = []
    if media_type == "movie" and year:
        candidates.append(f"{title} ({year} film)")
    if media_type == "tv":
        candidates.append(f"{title} (TV series)")
    candidates.append(title)

    for candidate in candidates:
        page = _load_page(candidate)
        if page is None:
            continue
        text = _page_text(page)
        if text:
            return page.title, text

    # Last resort: ask Wikipedia's own search for the closest matching title.
    try:
        search_results = wikipedia.search(title, results=1)
    except Exception:
        search_results = []

    if search_results:
        page = _load_page(search_results[0])
        if page is not None:
            text = _page_text(page)
            if text:
                return page.title, text

    return None, None


def load_existing_output():
    if not os.path.exists(OUTPUT_PATH):
        return {}
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {r["uid"]: r for r in records}


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        source_records = json.load(f)

    done = load_existing_output()
    print(f"{len(done)} titles already have plots, {len(source_records) - len(done)} left to do.")

    found_count = 0
    missing_count = 0

    for i, record in enumerate(source_records, start=1):
        if record["uid"] in done:
            continue

        try:
            wiki_title, plot = find_plot(record["title"], record["year"], record["media_type"])
        except Exception as e:
            print(f"  unexpected error on '{record['title']}': {e}")
            wiki_title, plot = None, None

        record = dict(record)
        record["wiki_title"] = wiki_title
        record["plot"] = plot or ""

        if plot:
            found_count += 1
        else:
            missing_count += 1

        done[record["uid"]] = record

        if i % SAVE_EVERY == 0:
            save(done)
            print(f"  processed {i}/{len(source_records)} "
                  f"(found: {found_count}, missing: {missing_count})")

        time.sleep(0.1)  # be polite to Wikipedia's servers

    save(done)
    print(f"\nDone. Plots found for {found_count} titles, "
          f"missing for {missing_count}. Saved to {OUTPUT_PATH}")


def save(done_dict):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(list(done_dict.values()), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
