"""
Free, local alternative to llm_agent.py: the same tool-use loop, but
talking to a model running on your own machine via Ollama instead of the
paid Claude API. No API key, no per-token cost - the tradeoff is a much
smaller, less capable model, so tool-calling is noticeably less
reliable (it may pick the wrong tool, mangle arguments, or just answer
directly from its own training data without calling a tool at all).

Requires Ollama running locally:
    brew services start ollama
    ollama pull llama3.1:8b

Reuses tools.py's TOOL_FUNCTIONS, the evidence builders, and the
confidence/finalize logic from llm_agent.py completely unchanged - only
the API call format and tool-schema shape differ between providers.
"""

import ollama

from llm_agent import (
    EVIDENCE_BUILDERS,
    SYSTEM_PROMPT,
    TOOL_FUNCTIONS,
    confidence_ok,
    finalize_response,
)

MODEL = "llama3.1:8b"
MAX_ROUNDS = 4

# Same 4 tools as llm_agent.py, just in Ollama's schema shape
# ({"type": "function", "function": {...}}) instead of Claude's.
OLLAMA_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_titles",
            "description": "Look up a title by name or keyword. Use for direct lookups "
                            "like 'what is X about' or finding a specific show/movie.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Title or keywords to search for"},
                    "top_n": {"type": "integer", "description": "Max results to return"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_by_constraints",
            "description": "Filter the catalog by structured constraints only (genre, "
                            "media type, year range, max runtime, min rating), with no "
                            "text query or ranking. Use for requests like 'crime movies "
                            "from the 2010s rated above 7'.",
            "parameters": {
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
    },
    {
        "type": "function",
        "function": {
            "name": "compare_titles",
            "description": "Look up two titles and return a structured comparison: "
                            "rating, runtime, year, shared/distinct genres, and each "
                            "title's plot text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title_a": {"type": "string"},
                    "title_b": {"type": "string"},
                },
                "required": ["title_a", "title_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend",
            "description": "Get ranked recommendations for a free-text description of "
                            "what the user wants to watch, using hybrid semantic + "
                            "keyword retrieval and a weighted ranking formula. Optionally "
                            "narrow by genres, max runtime, or min rating.",
            "parameters": {
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
    },
]


def handle_query(query, max_rounds=MAX_ROUNDS):
    """Same shape/behavior as llm_agent.handle_query(), routed through a
    local Ollama model instead of Claude."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    tool_calls_made = []
    results_by_tool = {}
    all_evidence = []

    for _round in range(max_rounds):
        response = ollama.chat(model=MODEL, messages=messages, tools=OLLAMA_TOOL_SCHEMAS)

        if not response.message.tool_calls:
            return finalize_response(tool_calls_made, results_by_tool, all_evidence,
                                      response.message.content or "")

        messages.append({
            "role": "assistant",
            "content": response.message.content,
            "tool_calls": response.message.tool_calls,
        })

        for tool_call in response.message.tool_calls:
            name = tool_call.function.name
            args = dict(tool_call.function.arguments)

            if name not in TOOL_FUNCTIONS:
                messages.append({"role": "tool", "tool_name": name,
                                  "content": f"Unknown tool '{name}'"})
                continue

            try:
                result = TOOL_FUNCTIONS[name](**args)
            except Exception as e:
                messages.append({"role": "tool", "tool_name": name,
                                  "content": f"Error calling {name}: {e}"})
                continue

            tool_calls_made.append((name, args))
            results_by_tool[name] = result

            if not confidence_ok(name, result):
                return finalize_response(tool_calls_made, results_by_tool, all_evidence,
                                          None, low_confidence=True)

            all_evidence.extend(EVIDENCE_BUILDERS[name](result))
            messages.append({"role": "tool", "tool_name": name, "content": str(result)[:4000]})

    return finalize_response(tool_calls_made, results_by_tool, all_evidence,
                              "Reached the tool-call limit without a final answer.")
