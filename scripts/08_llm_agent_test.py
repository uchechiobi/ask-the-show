"""
Step 8: Test the LLM-routed agent (Claude decides which tools to call).

Requires ANTHROPIC_API_KEY in .env - get one at
https://console.anthropic.com/settings/keys. This calls the real Claude
API, which costs a small amount per query (a few Haiku queries here
should cost well under a cent).
"""

import llm_agent

DEMO_QUERIES = [
    "compare The Sopranos and The Wire",
    "recommend a detective show about a murder in a small town",
    "what is The Wire about",
    "something like Alien but shorter",
    "something good",  # deliberately vague - should trigger the confidence guardrail
]


def print_result(query, result):
    print(f"Query: {query!r}")
    print(f"  tool calls: {result['tool_calls']}")
    print(f"  low_confidence: {result['low_confidence']}")
    print()
    print(result["response"])
    print("\n" + "=" * 70 + "\n")


def main():
    print(f"Running {len(DEMO_QUERIES)} demo queries through the LLM-routed agent...\n")
    for query in DEMO_QUERIES:
        result = llm_agent.handle_query(query)
        print_result(query, result)

    print("Now try your own queries.\n")
    while True:
        query = input("Query (or 'quit'): ").strip()
        if query.lower() in ("quit", "exit", ""):
            break
        result = llm_agent.handle_query(query)
        print()
        print_result(query, result)


if __name__ == "__main__":
    main()
