"""
Step 7: Test the Day 4 guardrails.

Runs a normal comparison and recommendation query (to see evidence
display working) plus two deliberately vague/unanswerable queries (to
see the confidence guardrail actually trigger), then directly tests the
title-validation guardrail with a fabricated fake title - since nothing
in this system generates free text today, every title already comes
straight from the dataset by construction, so that guardrail can't be
triggered through normal use. Feeding it a fake title by hand proves the
check genuinely works rather than being an inert stub.
"""

import agent
import guardrails

DEMO_QUERIES = [
    ("Comparison (evidence display, high confidence)", "compare The Sopranos and The Wire"),
    ("Recommendation (evidence + scores)", "recommend a detective show about a murder in a small town"),
    ("Vague query (SHOULD trigger the confidence guardrail)", "something good"),
    ("Nonsense query (SHOULD trigger the confidence guardrail)", "asdkjaslkdj random xyz"),
]


def run_demo_queries():
    for label, query in DEMO_QUERIES:
        print(f"[{label}]")
        print(f"Query: {query!r}")
        result = agent.handle_query(query)
        print(f"  low_confidence: {result['low_confidence']}")
        print()
        print(result["response"])
        print("\n" + "=" * 70 + "\n")


def demo_title_validation():
    print("[Direct test of the title-validation guardrail]")
    print("(feeding it a fabricated title by hand, since nothing in this "
          "system generates free text that could hallucinate one on its own)\n")
    fake_evidence = [
        guardrails.build_evidence("The Wire", "title_match", "a real title in the dataset", None),
        guardrails.build_evidence("The Totally Fake Show That Does Not Exist", "hallucinated", "made up", 0.9),
    ]
    clean, flagged = guardrails.validate_titles(fake_evidence)
    print(f"  input titles: {[e['title'] for e in fake_evidence]}")
    print(f"  passed validation: {[e['title'] for e in clean]}")
    print(f"  flagged as not in dataset: {flagged}")
    print("\n" + "=" * 70 + "\n")


def main():
    run_demo_queries()
    demo_title_validation()

    print("Now try your own queries - try something vague to trigger the guardrail.\n")
    while True:
        query = input("Query (or 'quit'): ").strip()
        if query.lower() in ("quit", "exit", ""):
            break
        result = agent.handle_query(query)
        print(f"\nlow_confidence: {result['low_confidence']}\n")
        print(result["response"])
        print()


if __name__ == "__main__":
    main()
