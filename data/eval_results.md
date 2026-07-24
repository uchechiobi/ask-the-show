# Ask the Show - Day 5 Evaluation

20 questions, hybrid (semantic + BM25) vs. vector-only retrieval, top-5.

| Query type | N | Hybrid | Vector-only |
|---|---|---|---|
| Lookup | 5 | 100% | 100% |
| Comparison | 2 | 100% | 100% |
| Filter | 1 | 100% | 100% |
| Recommend | 7 | 100% | 57% |
| **Overall (retrieval, P@5)** | 15 | 100% | 80% |
| Guardrail correctly triggered (vague queries) | 5 | 100% | 100% |

_Lookup, comparison, and filter questions don't use semantic/BM25 search at all, so they score identically in both columns by design - only Recommend and the guardrail check are actually retrieval-method-dependent._
