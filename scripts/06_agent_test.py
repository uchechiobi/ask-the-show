"""
Step 6: Test the agent layer.

Runs four example queries - one of each type (comparison, constrained
recommendation, simple lookup, mixed) - so you can see the rule-based
router picking a different path for each, then drops into an
interactive loop so you can try your own.
"""

import agent

DEMO_QUERIES = [
    ("Comparison", "compare The Sopranos and The Wire"),
    ("Constrained recommendation", "recommend a mind-bending science fiction movie under 120 minutes"),
    ("Simple lookup", "what is The Wire about"),
    ("Mixed (lookup + recommend)", "something like Alien but shorter"),
    ("Pure filter (no ranking)", "crime movies from the 2010s rated above 7"),
]


def print_result(query, result):
    print(f"Query: {query!r}")
    print(f"  -> intent: {result['intent']}")
    print(f"  -> reason: {result['reason']}")
    print(f"  -> tool calls: {[name for name, _args in result['tool_calls']]}")
    print()
    print(result["response"])
    print("\n" + "-" * 70 + "\n")


def main():
    print(f"Running {len(DEMO_QUERIES)} demo queries to show the router picking different tools...\n")
    print("=" * 70 + "\n")
    for label, query in DEMO_QUERIES:
        print(f"[{label}]")
        result = agent.handle_query(query)
        print_result(query, result)

    print("Now try your own queries.\n")
    while True:
        query = input("Query (or 'quit'): ").strip()
        if query.lower() in ("quit", "exit", ""):
            break
        result = agent.handle_query(query)
        print()
        print_result(query, result)


if __name__ == "__main__":
    main()
