"""
main.py
-------
Entry point. Builds the index (if needed) then starts the CLI agent.
Usage: python src/main.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from retriever import Retriever
from agent import answer

INDEX_PATH = "data/index.faiss"


def setup():
    """Build the document index if it doesn't exist yet."""
    if not os.path.exists(INDEX_PATH):
        print("[Setup] No index found. Building from sample documents...")
        r = Retriever()
        r.build_index()
    else:
        print("[Setup] Index found. Ready.")


def run_cli():
    print("\n" + "=" * 60)
    print("  Digital Twin Q&A Agent - SmartFactory GmbH")
    print("  Type 'exit' to quit")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break
        if not query:
            continue

        print("\nAgent: ", end="", flush=True)
        result = answer(query)
        print(result["answer"])
        if result["sources"]:
            print(f"\n  Sources: {', '.join(result['sources'])}")
        print(f"  Tools used: {', '.join(result['tools_used'])}\n")


if __name__ == "__main__":
    setup()
    run_cli()
