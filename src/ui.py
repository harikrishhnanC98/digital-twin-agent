"""
ui.py
-----
Streamlit web interface for the Digital Twin Q&A Agent.
Run with: streamlit run src/ui.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from agent import answer
from retriever import Retriever

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Digital Twin Q&A Agent",
    page_icon="🏭",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🏭 Digital Twin Agent")
    st.caption("SmartFactory - Smart Factory Assistant")
    st.divider()

    st.subheader("Setup")
    if st.button("Build / Rebuild Index", use_container_width=True):
        with st.spinner("Indexing documents..."):
            try:
                r = Retriever()
                r.build_index()
                st.success("Index built successfully.")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    st.subheader("Example questions")
    examples = [
        "What is the maintenance schedule for conveyor belt A3?",
        "What is the current vibration level of conveyor A3?",
        "When was robot arm 7 last serviced?",
        "What are the safety procedures for the welding station?",
        "Explain the digital twin data model used in the system.",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state["query_input"] = ex

    st.divider()
    st.caption("Running locally via Ollama · No data leaves the factory network")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Smart Factory Q&A")
st.caption("Ask questions about your factory equipment, maintenance history, and technical documentation.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"Sources: {', '.join(msg['sources'])}")

# Input
query = st.chat_input("Ask about your factory...", key="chat_input")

# Handle example button click
if "query_input" in st.session_state:
    query = st.session_state.pop("query_input")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating answer..."):
            result = answer(query)

        st.markdown(result["answer"])

        if result["sources"]:
            st.caption(f"Sources: {', '.join(result['sources'])}")

        with st.expander("Debug info"):
            st.json({
                "tools_used": result["tools_used"],
                "sources": result["sources"],
            })

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
