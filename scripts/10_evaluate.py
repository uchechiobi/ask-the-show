"""
Step 10: Evaluate the system on data/eval_questions.json, twice - once
with hybrid retrieval (semantic + BM25), once with vector-only retrieval
(semantic alone, via recommend()'s use_bm25=False ablation added for
this purpose). Everything else (ranking formula, weights, constraints)
stays identical between the two runs, so any difference in results is
attributable to the retrieval method alone.

This calls tools.py directly, not agent.py's router - Day 5 measures
retrieval quality, not routing accuracy (that was Day 3/4's job).

lookup / comparison / filter questions don't go through semantic or
BM25 search at all (title_match, direct lookup, and structural filtering
respectively), so they score identically in both runs - that's expected,
not a bug, and the report says so explicitly.

Outputs:
    data/eval_results.json  - full raw results, machine-readable
    data/eval_results.md    - Markdown table, paste straight into a README
    data/eval_results.png   - bar chart, screenshot-friendly
"""

import json
import os

import guardrails
import tools

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval_questions.json")
RESULTS_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval_results.json")
RESULTS_MD_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval_results.md")
RESULTS_PNG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval_results.png")

TOP_K = 5


def run_lookup(q):
    result = tools.search_titles(q["query"], top_n=TOP_K)
    titles = [r["title"] for r in result["results"]]
    hit = any(t in titles for t in q["expected_titles"])
    return hit, titles


def run_comparison(q):
    result = tools.compare_titles(q["title_a"], q["title_b"])
    if "error" in result:
        return False, []
    resolved = [result["title_a"]["title"], result["title_b"]["title"]]
    hit = set(q["expected_titles"]) == set(resolved)
    return hit, resolved


def run_filter(q):
    result = tools.filter_by_constraints(
        genres=q.get("genres"), media_type=q.get("media_type"),
        year_min=q.get("year_min"), year_max=q.get("year_max"),
        max_runtime=q.get("max_runtime"), min_rating=q.get("min_rating"),
        top_n=TOP_K,
    )
    titles = [r["title"] for r in result["results"]]
    hit = any(t in titles for t in q["expected_titles"])
    return hit, titles


def run_recommend(q, use_bm25):
    result = tools.recommend(
        q["query"], genres=q.get("genres"), max_runtime=q.get("max_runtime"),
        min_rating=q.get("min_rating"), top_n=TOP_K, use_bm25=use_bm25,
    )
    titles = [r["title"] for r in result["results"]]
    hit = any(t in titles for t in q["expected_titles"])
    return hit, titles


def run_vague(q, use_bm25):
    result = tools.recommend(q["query"], top_n=TOP_K, use_bm25=use_bm25)
    confident = guardrails.has_sufficient_confidence(
        result.get("top_semantic_raw"), result.get("top_bm25_raw")
    )
    # "hit" here means the guardrail correctly recognized this as unanswerable.
    return not confident, []


def evaluate_question(q, use_bm25):
    qtype = q["query_type"]
    if qtype == "lookup":
        hit, titles = run_lookup(q)
    elif qtype == "comparison":
        hit, titles = run_comparison(q)
    elif qtype == "filter":
        hit, titles = run_filter(q)
    elif qtype == "recommend":
        hit, titles = run_recommend(q, use_bm25)
    elif qtype == "vague":
        hit, titles = run_vague(q, use_bm25)
    else:
        raise ValueError(f"Unknown query_type: {qtype}")
    return hit, titles


def build_summary(rows):
    """Precision@K per query_type per mode, for the retrieval-driven
    types; a separate guardrail hit-rate for the vague type."""
    retrieval_types = ["lookup", "comparison", "filter", "recommend"]
    summary = {}
    for qtype in retrieval_types:
        type_rows = [r for r in rows if r["query_type"] == qtype]
        if not type_rows:
            continue
        summary[qtype] = {
            "n": len(type_rows),
            "hybrid_precision": sum(r["hybrid_hit"] for r in type_rows) / len(type_rows),
            "vector_precision": sum(r["vector_hit"] for r in type_rows) / len(type_rows),
        }

    retrieval_rows = [r for r in rows if r["query_type"] in retrieval_types]
    summary["overall_retrieval"] = {
        "n": len(retrieval_rows),
        "hybrid_precision": sum(r["hybrid_hit"] for r in retrieval_rows) / len(retrieval_rows),
        "vector_precision": sum(r["vector_hit"] for r in retrieval_rows) / len(retrieval_rows),
    }

    vague_rows = [r for r in rows if r["query_type"] == "vague"]
    summary["guardrail_vague"] = {
        "n": len(vague_rows),
        "hybrid_precision": sum(r["hybrid_hit"] for r in vague_rows) / len(vague_rows),
        "vector_precision": sum(r["vector_hit"] for r in vague_rows) / len(vague_rows),
    }
    return summary


def print_console_table(summary):
    label_map = {
        "lookup": "Lookup", "comparison": "Comparison", "filter": "Filter",
        "recommend": "Recommend", "overall_retrieval": "OVERALL (retrieval)",
        "guardrail_vague": "Guardrail (vague queries)",
    }
    print(f"{'Query type':<26}{'N':>4}{'Hybrid':>10}{'Vector-only':>14}")
    print("-" * 54)
    for key in ["lookup", "comparison", "filter", "recommend", "overall_retrieval", "guardrail_vague"]:
        s = summary[key]
        print(f"{label_map[key]:<26}{s['n']:>4}{s['hybrid_precision']*100:>9.0f}%"
              f"{s['vector_precision']*100:>13.0f}%")


def save_markdown(summary, rows):
    label_map = {
        "lookup": "Lookup", "comparison": "Comparison", "filter": "Filter",
        "recommend": "Recommend", "overall_retrieval": "**Overall (retrieval, P@5)**",
        "guardrail_vague": "Guardrail correctly triggered (vague queries)",
    }
    lines = [
        "# Ask the Show - Day 5 Evaluation",
        "",
        f"{len(rows)} questions, hybrid (semantic + BM25) vs. vector-only retrieval, top-{TOP_K}.",
        "",
        "| Query type | N | Hybrid | Vector-only |",
        "|---|---|---|---|",
    ]
    for key in ["lookup", "comparison", "filter", "recommend", "overall_retrieval", "guardrail_vague"]:
        s = summary[key]
        lines.append(f"| {label_map[key]} | {s['n']} | {s['hybrid_precision']*100:.0f}% | "
                      f"{s['vector_precision']*100:.0f}% |")
    lines.append("")
    lines.append("_Lookup, comparison, and filter questions don't use semantic/BM25 search "
                  "at all, so they score identically in both columns by design - only "
                  "Recommend and the guardrail check are actually retrieval-method-dependent._")

    with open(RESULTS_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_chart(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ["lookup", "comparison", "filter", "recommend", "overall_retrieval", "guardrail_vague"]
    labels = ["Lookup", "Comparison", "Filter", "Recommend", "Overall\n(retrieval)", "Guardrail\n(vague)"]
    hybrid_vals = [summary[k]["hybrid_precision"] * 100 for k in keys]
    vector_vals = [summary[k]["vector_precision"] * 100 for k in keys]

    x = range(len(keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], hybrid_vals, width, label="Hybrid (semantic + BM25)")
    ax.bar([i + width / 2 for i in x], vector_vals, width, label="Vector-only")
    ax.set_ylabel(f"Precision@{TOP_K} (%)")
    ax.set_title("Ask the Show - Day 5 Evaluation: Hybrid vs. Vector-only Retrieval")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_PNG_PATH, dpi=150)


def question_label(q):
    if q.get("query"):
        return q["query"]
    if q["query_type"] == "comparison":
        return f"compare {q['title_a']} vs {q['title_b']}"
    if q["query_type"] == "filter":
        parts = [f"{k}={v}" for k, v in q.items()
                 if k not in ("id", "query_type", "expected_titles", "notes") and v]
        return "filter: " + ", ".join(parts)
    return f"question {q['id']}"


def main():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Evaluating {len(questions)} questions (top-{TOP_K}), hybrid vs. vector-only...\n")

    rows = []
    for q in questions:
        hybrid_hit, hybrid_titles = evaluate_question(q, use_bm25=True)
        vector_hit, vector_titles = evaluate_question(q, use_bm25=False)
        rows.append({
            "id": q["id"], "query_type": q["query_type"],
            "query": question_label(q),
            "expected_titles": q["expected_titles"],
            "hybrid_hit": hybrid_hit, "hybrid_titles": hybrid_titles,
            "vector_hit": vector_hit, "vector_titles": vector_titles,
        })
        mark = lambda b: "PASS" if b else "FAIL"  # noqa: E731
        print(f"[{q['id']:>2}] {q['query_type']:<10} hybrid={mark(hybrid_hit):<4} "
              f"vector={mark(vector_hit):<4}  {rows[-1]['query'][:55]}")

    summary = build_summary(rows)

    print()
    print_console_table(summary)

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=2, ensure_ascii=False)
    save_markdown(summary, rows)
    save_chart(summary)

    print(f"\nSaved: {RESULTS_JSON_PATH}\nSaved: {RESULTS_MD_PATH}\nSaved: {RESULTS_PNG_PATH}")


if __name__ == "__main__":
    main()
