"""
agent.py
--------
The agent orchestrator.

Decision flow:
  1. Classify the user query to decide which tool(s) to call
  2. Call the tool(s) and collect context
  3. Build a prompt with the retrieved context
  4. Call the local LLM (via Ollama) to generate the final answer
  5. Return the answer with source citations

Design principle: the agent never answers from memory alone.
Every response is grounded in retrieved context (RAG) or live data.
This is critical for a factory setting where outdated or hallucinated
information could lead to incorrect maintenance decisions.
"""

import json
import re
from typing import List, Tuple

import requests

from tools import TOOLS, tool_search_knowledge_base

# ---------------------------------------------------------------------------
# LLM configuration — Ollama (local, on-premise)
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3:mini"   # swap for mistral, llama3.2, etc.
TEMPERATURE = 0.2     # low temp = factual, deterministic answers


# ---------------------------------------------------------------------------
# Query classifier
# ---------------------------------------------------------------------------

def classify_query(query: str) -> List[str]:
    """
    Decide which tools to invoke based on keywords in the query.
    Returns a list of tool names to call (in order).

    In a more advanced version this classification step itself
    could be handled by the LLM (ReAct pattern). For reliability
    and speed in a factory environment, rule-based classification
    is preferred at this stage.
    """
    query_lower = query.lower()
    selected = []

    for tool_name, meta in TOOLS.items():
        if any(kw in query_lower for kw in meta["keywords"]):
            selected.append(tool_name)

    # Always include knowledge base search as a fallback
    if not selected or "search_knowledge_base" not in selected:
        selected.insert(0, "search_knowledge_base")

    return selected


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_context(query: str, tool_names: List[str]) -> Tuple[str, List[str]]:
    """
    Call the selected tools and assemble their outputs into a
    context block for the LLM prompt.
    Returns (context_text, list_of_sources).
    """
    context_parts = []
    sources = []

    for tool_name in tool_names:
        tool = TOOLS[tool_name]
        fn = tool["fn"]

        if tool_name == "search_knowledge_base":
            result = fn(query=query, top_k=5)
            for item in result["results"]:
                context_parts.append(f"[Document: {item['source']}]\n{item['text']}")
                if item["source"] not in sources:
                    sources.append(item["source"])

        elif tool_name == "get_sensor_data":
            # Extract machine_id and metric from query heuristically
            machine_id = extract_machine_id(query)
            metric = extract_metric(query)
            result = fn(machine_id=machine_id, metric=metric)
            context_parts.append(f"[Sensor Data: {result['machine']}]\n"
                                  f"Metric: {result['metric']}, "
                                  f"Value: {result.get('value')} {result.get('unit','')}, "
                                  f"Status: {result.get('status')}")
            sources.append(f"Live sensor: {result['machine']}")

        elif tool_name == "get_maintenance_log":
            machine_id = extract_machine_id(query)
            result = fn(machine_id=machine_id)
            for event in result["events"]:
                context_parts.append(f"[Maintenance Log: {result['machine']}]\n"
                                      f"Date: {event['date']}, Type: {event['type']}\n"
                                      f"Notes: {event['notes']}")
            sources.append(f"Maintenance log: {result['machine']}")

    context = "\n\n".join(context_parts)
    return context, sources


def extract_machine_id(query: str) -> str:
    """Simple heuristic — look for known machine identifiers."""
    patterns = ["conveyor_a3", "conveyor a3", "robot_arm_7", "robot arm 7"]
    q = query.lower()
    for p in patterns:
        if p in q:
            return p.replace(" ", "_")
    return "conveyor_A3"   # default for demo


def extract_metric(query: str) -> str:
    """Simple heuristic — look for known metric names."""
    for metric in ["temperature", "vibration", "cycle_time", "speed", "pressure"]:
        if metric in query.lower():
            return metric
    return "temperature"   # default


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> str:
    """
    Send prompt to local Ollama instance and return the generated text.
    Streams the response and concatenates.
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "temperature": TEMPERATURE, "stream": True},
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
        output = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                output += chunk.get("response", "")
                if chunk.get("done"):
                    break
        return output.strip()
    except requests.exceptions.ConnectionError:
        return (
            "[Error] Cannot reach Ollama. Make sure Ollama is running: `ollama serve`\n"
            f"Then pull the model: `ollama pull {MODEL}`"
        )
    except Exception as e:
        return f"[Error] LLM call failed: {e}"


# ---------------------------------------------------------------------------
# Main agent entry point
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a smart factory assistant for SmartFactory GmbH.
You help engineers and operators understand their production equipment,
digital twin data, maintenance history and technical documentation.

Rules:
- Answer only based on the provided context. Do not make up facts.
- If the context does not contain enough information, say so clearly.
- Be concise and precise. Use technical language appropriate for engineers.
- Always cite the source document or data source at the end of your answer.
"""


def answer(query: str) -> dict:
    """
    Main agent function. Takes a user query and returns a structured response.

    Returns:
        {
            "answer": str,
            "sources": List[str],
            "tools_used": List[str],
            "query": str,
        }
    """
    # Step 1: classify
    tool_names = classify_query(query)

    # Step 2: retrieve context
    context, sources = build_context(query, tool_names)

    # Step 3: build prompt
    prompt = f"""{SYSTEM_PROMPT}

--- Context ---
{context}
--- End of Context ---

Question: {query}

Answer:"""

    # Step 4: generate
    llm_answer = call_llm(prompt)

    return {
        "query": query,
        "answer": llm_answer,
        "sources": sources,
        "tools_used": tool_names,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Digital Twin Q&A Agent — type 'exit' to quit\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        result = answer(query)
        print(f"\nAgent: {result['answer']}")
        print(f"\nSources: {', '.join(result['sources']) or 'none'}")
        print(f"Tools used: {', '.join(result['tools_used'])}\n")
