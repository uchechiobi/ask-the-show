"""
Step 9: Test the free, local LLM-routed agent (Ollama + Llama 3.1 8B
deciding which tools to call, instead of the paid Claude API).

Requires Ollama running locally with the model pulled:
    brew services start ollama
    ollama pull llama3.1:8b

Expect this to be visibly less reliable than 08_llm_agent_test.py's
Claude version - an 8B local model is much weaker at picking the right
tool and filling in arguments correctly than Claude is. That gap is the
real, honest tradeoff for running for free on your own machine.
"""

import local_llm_agent

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
    print(f"Running {len(DEMO_QUERIES)} demo queries through the local LLM-routed agent...\n")
    for query in DEMO_QUERIES:
        result = local_llm_agent.handle_query(query)
        print_result(query, result)

    print("Now try your own queries.\n")
    while True:
        query = input("Query (or 'quit'): ").strip()
        if query.lower() in ("quit", "exit", ""):
            break
        result = local_llm_agent.handle_query(query)
        print()
        print_result(query, result)


if __name__ == "__main__":
    main()
