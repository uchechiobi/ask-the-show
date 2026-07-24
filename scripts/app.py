"""
Day 6: Streamlit chat UI for Ask the Show.

Wired to agent.py's rule-based router from Day 3/4 - free, local, no
API keys required. Every answer shows its sources (title, retrieval
method, score, excerpt) always visible below the response, never in a
collapsed expander. Low-confidence answers (the Day 4 guardrail) render
through st.warning() so declining to answer looks visually distinct
from a normal response.

Run locally:   streamlit run scripts/app.py   (from the project root)
Deploy:        see the deployment walkthrough in the chat / README.
"""

import os
import sys

# Make sibling modules (agent, tools, guardrails, ...) importable
# regardless of how the entrypoint is invoked (plain python, streamlit
# run from repo root, or Streamlit Community Cloud's runner).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

import agent
import tools

st.set_page_config(page_title="Ask the Show", layout="centered")
st.title("Ask the Show")
st.caption(
    "An explainable AI media-discovery agent. Every answer shows the exact "
    "evidence that backed it - title, retrieval method, score, and excerpt."
)


@st.cache_resource(show_spinner="Loading the catalog, embeddings, and BM25 index (first run only)...")
def warm_up():
    """Force tools.py's lazy caches (records, embedding model, Chroma
    collection, BM25 index) to load once per app process, with a visible
    spinner instead of a silent multi-second freeze on the first query."""
    tools.known_genres()
    tools.recommend("warm up", top_n=1)
    return True


warm_up()

if "messages" not in st.session_state:
    st.session_state.messages = []


def to_display(result):
    """Extract exactly what the UI needs from an agent.handle_query()
    result, in a shape that's reused for both the live response and
    history replay after a rerun."""
    if result["low_confidence"]:
        headline = result["response"]
    else:
        # finalize() appends the evidence block as text onto "response" -
        # strip it back off since we render evidence as separate UI
        # elements instead of duplicating it inline as text too.
        headline = result["response"].split("\n\nEvidence:")[0]
    return {
        "low_confidence": result["low_confidence"],
        "headline": headline,
        "evidence": result["evidence"],
        "flagged_titles": result["flagged_titles"],
        "intent": result["intent"],
        "reason": result["reason"],
        "tool_calls": result["tool_calls"],
    }


def render_assistant(display):
    if display["low_confidence"]:
        st.warning(display["headline"])
    else:
        st.markdown(display["headline"])

        if display["evidence"]:
            st.markdown(f"**Sources ({len(display['evidence'])})**")
            for e in display["evidence"]:
                with st.container(border=True):
                    score_str = f"{e['score']:.3f}" if isinstance(e["score"], (int, float)) else "n/a"
                    st.markdown(f"**{e['title']}**")
                    st.caption(f"source: {e['source']}  ·  score: {score_str}")
                    if e["excerpt"]:
                        st.caption(e["excerpt"])

        if display["flagged_titles"]:
            st.caption(f"Flagged and removed (not in dataset): {', '.join(display['flagged_titles'])}")

        with st.expander("Routing details"):
            st.markdown(f"**Intent:** {display['intent']}")
            st.markdown(f"**Reason:** {display['reason']}")
            st.markdown(f"**Tool calls:** {display['tool_calls']}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant(msg["display"])

query = st.chat_input("Ask about a show or movie...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            result = agent.handle_query(query)
            display = to_display(result)
        except Exception as e:
            st.error(f"Something went wrong handling that query: {e}")
            display = None

        if display:
            render_assistant(display)
            st.session_state.messages.append({"role": "assistant", "display": display})
