"""
LLM-routed agent layer - an alternative to agent.py's rule-based router.

Same 4 tools from tools.py, but now Claude itself reads the query,
decides which tool(s) to call and with what arguments, and writes the
final response - instead of a fixed list of regex patterns.

Tradeoff versus agent.py: more flexible on phrasing the regex router
never anticipated, but the routing decision itself is no longer
something you can read in source code - you're trusting Claude's
tool-call choices rather than an inspectable "reason" string. It also
costs a small amount of money per query and needs an ANTHROPIC_API_KEY.

Because Claude genuinely writes free text here (unlike agent.py's
templated responses), the Day 4 guardrails matter for real:
  - The confidence check runs BEFORE Claude writes anything. If a
    retrieval tool's scores were too weak, we return the "not confident"
    message without asking Claude to answer at all - so it's never
    tempted to fall back on its own training knowledge about a title.
  - After Claude answers, we scan its actual response text for any
    dataset title it mentions that wasn't part of the evidence its own
    tool calls returned, and flag/strip it.

This is a shared module - other scripts import it, it isn't run directly.
"""

import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv

import guardrails
import tools
from agent import (
    evidence_for_comparison,
    evidence_for_filter,
    evidence_for_lookup,
    evidence_for_recommend,
)

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
MAX_ROUNDS = 4

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = (
    "You are the Ask the Show agent. You help users search, filter, compare, and "
    "get recommendations for movies and TV shows using ONLY the tools provided - "
    "never your own general knowledge about titles. Every title you mention in "
    "your response MUST come from a tool result you actually received. If the "
    "tools return nothing relevant, say so plainly instead of guessing. Keep "
    "responses concise: a short answer plus the key facts (rating, genres, why "
    "it matches) for each title mentioned."
)

TOOL_SCHEMAS = [
    {
        "name": "search_titles",
        "description": "Look up a title by name or keyword. Use for direct lookups "
                        "like 'what is X about' or finding a specific show/movie.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Title or keywords to search for"},
                "top_n": {"type": "integer", "description": "Max results to return"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "filter_by_constraints",
        "description": "Filter the catalog by structured constraints only (genre, "
                        "media type, year range, max runtime, min rating), with no "
                        "text query or ranking. Use for requests like 'crime movies "
                        "from the 2010s rated above 7'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "genres": {"type": "array", "items": {"type": "string"}},
                "media_type": {"type": "string", "enum": ["movie", "tv"]},
                "year_min": {"type": "integer"},
                "year_max": {"type": "integer"},
                "max_runtime": {"type": "integer", "description": "Max runtime in minutes"},
                "min_rating": {"type": "number"},
                "top_n": {"type": "integer"},
            },
        },
    },
    {
        "name": "compare_titles",
        "description": "Look up two titles and return a structured comparison: "
                        "rating, runtime, year, shared/distinct genres, and each "
                        "title's plot text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title_a": {"type": "string"},
                "title_b": {"type": "string"},
            },
            "required": ["title_a", "title_b"],
        },
    },
    {
        "name": "recommend",
        "description": "Get ranked recommendations for a free-text description of "
                        "what the user wants to watch, using hybrid semantic + "
                        "keyword retrieval and a weighted ranking formula. Optionally "
                        "narrow by genres, max runtime, or min rating.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Description of what the user wants to watch"},
                "genres": {"type": "array", "items": {"type": "string"}},
                "max_runtime": {"type": "integer"},
                "min_rating": {"type": "number"},
                "top_n": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
]

TOOL_FUNCTIONS = {
    "search_titles": tools.search_titles,
    "filter_by_constraints": tools.filter_by_constraints,
    "compare_titles": tools.compare_titles,
    "recommend": tools.recommend,
}

EVIDENCE_BUILDERS = {
    "compare_titles": evidence_for_comparison,
    "recommend": evidence_for_recommend,
    "filter_by_constraints": evidence_for_filter,
    "search_titles": evidence_for_lookup,
}


def confidence_ok(name, result):
    """Only recommend() and search_titles' keyword-search path produce a
    retrieval score worth gating on; the other tools are deterministic
    lookups with nothing to be unconfident about."""
    if name == "recommend":
        return guardrails.has_sufficient_confidence(
            result.get("top_semantic_raw"), result.get("top_bm25_raw")
        )
    if name == "search_titles" and result.get("method") == "keyword_search":
        top_score = result["results"][0]["match_score"] if result["results"] else 0.0
        return (top_score or 0.0) >= guardrails.CONFIDENCE_BM25_THRESHOLD
    return True


def handle_query(query, max_rounds=MAX_ROUNDS):
    """
    Route a query through Claude's tool-use loop. Returns a dict shaped
    like agent.handle_query()'s output, for direct comparison:
        tool_calls      - [(tool_name, arguments), ...] Claude actually made
        result          - {tool_name: raw_result} for every call made
        response        - Claude's final text + evidence block, or the
                          low-confidence message
        evidence        - [{title, source, excerpt, score}, ...]
        flagged_titles  - titles mentioned without backing evidence
        low_confidence  - True if a retrieval tool's scores were too weak
    """
    messages = [{"role": "user", "content": query}]
    tool_calls_made = []
    results_by_tool = {}
    all_evidence = []

    for _round in range(max_rounds):
        response = client.messages.create(
            model=MODEL, max_tokens=1024, system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS, messages=messages,
        )

        if response.stop_reason != "tool_use":
            final_text = "".join(b.text for b in response.content if b.type == "text")
            return finalize_response(tool_calls_made, results_by_tool, all_evidence, final_text)

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            try:
                result = TOOL_FUNCTIONS[block.name](**block.input)
            except Exception as e:
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": f"Error calling {block.name}: {e}", "is_error": True,
                })
                continue

            tool_calls_made.append((block.name, block.input))
            results_by_tool[block.name] = result

            if not confidence_ok(block.name, result):
                return finalize_response(tool_calls_made, results_by_tool, all_evidence,
                                  None, low_confidence=True)

            all_evidence.extend(EVIDENCE_BUILDERS[block.name](result))
            tool_results.append({
                "type": "tool_result", "tool_use_id": block.id,
                "content": json.dumps(result, default=str)[:4000],
            })
        messages.append({"role": "user", "content": tool_results})

    return finalize_response(tool_calls_made, results_by_tool, all_evidence,
                      "Reached the tool-call limit without a final answer.")


def finalize_response(tool_calls, results_by_tool, evidence, response_text, low_confidence=False):
    if low_confidence:
        return {
            "tool_calls": tool_calls, "result": results_by_tool,
            "response": guardrails.LOW_CONFIDENCE_MESSAGE,
            "evidence": [], "flagged_titles": [], "low_confidence": True,
        }

    clean_evidence, flagged_dataset_titles = guardrails.validate_titles(evidence)
    evidence_titles = {e["title"] for e in clean_evidence}

    # The check that can actually fire now: does Claude's generated text
    # mention a dataset title it wasn't given as evidence for this query?
    mentioned = guardrails.scan_text_for_titles(response_text)
    unverified = [t for t in mentioned if t not in evidence_titles]
    for title in unverified:
        response_text = response_text.replace(title, f"[unverified: {title}]")

    response_text += "\n\n" + guardrails.render_evidence(clean_evidence)
    all_flags = flagged_dataset_titles + unverified
    if all_flags:
        response_text += f"\n\n[Note: flagged and removed - {', '.join(all_flags)}]"

    return {
        "tool_calls": tool_calls, "result": results_by_tool,
        "response": response_text,
        "evidence": clean_evidence,
        "flagged_titles": all_flags,
        "low_confidence": False,
    }
