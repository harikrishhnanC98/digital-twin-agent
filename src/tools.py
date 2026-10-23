"""
tools.py
--------
Tools the agent can call to answer a question.
Each tool is a plain Python function. The agent orchestrator decides
which tool(s) to invoke based on the user's query.

In a production deployment, tools would connect to live data sources:
the digital twin API, sensor databases, ERP systems, etc.
"""

from typing import List, Tuple
from retriever import Retriever

# Shared retriever instance (loaded once)
_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
        _retriever.load_index()
    return _retriever


# ---------------------------------------------------------------------------
# Tool: document retrieval (RAG)
# ---------------------------------------------------------------------------

def tool_search_knowledge_base(query: str, top_k: int = 5) -> dict:
    """
    Search the factory knowledge base for documents relevant to the query.
    Uses FAISS vector search over embedded document chunks.

    Returns:
        {
            "results": [{"text": ..., "source": ..., "score": ...}, ...],
            "count": int
        }
    """
    retriever = get_retriever()
    hits = retriever.retrieve(query, top_k=top_k)
    results = [
        {"text": text, "source": source, "score": round(1 / (1 + dist), 4)}
        for text, source, dist in hits
    ]
    return {"results": results, "count": len(results)}


# ---------------------------------------------------------------------------
# Tool: sensor data lookup (simulated)
# ---------------------------------------------------------------------------

def tool_get_sensor_data(machine_id: str, metric: str) -> dict:
    """
    Retrieve the latest sensor reading for a given machine and metric.
    In production: queries the digital twin's real-time data stream.
    Here: returns simulated data for demonstration.

    Args:
        machine_id: e.g. "conveyor_A3", "robot_arm_7"
        metric: e.g. "temperature", "vibration", "cycle_time"
    """
    # Simulated sensor readings — replace with real digital twin API call
    simulated = {
        ("conveyor_A3", "temperature"): {"value": 72.4, "unit": "°C", "status": "normal", "threshold": 90},
        ("conveyor_A3", "vibration"):   {"value": 3.1,  "unit": "mm/s", "status": "warning", "threshold": 3.0},
        ("robot_arm_7", "cycle_time"):  {"value": 4.8,  "unit": "s", "status": "normal", "threshold": 6.0},
        ("robot_arm_7", "temperature"): {"value": 58.2, "unit": "°C", "status": "normal", "threshold": 80},
    }
    key = (machine_id.lower(), metric.lower())
    if key in simulated:
        return {"machine": machine_id, "metric": metric, **simulated[key]}
    return {"machine": machine_id, "metric": metric, "value": None, "status": "unknown",
            "note": "No live data available for this machine/metric combination."}


# ---------------------------------------------------------------------------
# Tool: maintenance log lookup (simulated)
# ---------------------------------------------------------------------------

def tool_get_maintenance_log(machine_id: str, last_n: int = 3) -> dict:
    """
    Return the last N maintenance events for a machine.
    In production: queries the factory MES or ERP system.
    """
    simulated_logs = {
        "conveyor_A3": [
            {"date": "2024-09-15", "type": "Preventive", "technician": "M. Brandt", "notes": "Belt tension adjusted. No issues found."},
            {"date": "2024-07-02", "type": "Corrective",  "technician": "K. Müller", "notes": "Roller bearing replaced due to excess vibration."},
            {"date": "2024-04-10", "type": "Preventive", "technician": "M. Brandt", "notes": "Lubrication and inspection. Belt wear at 40%."},
        ],
        "robot_arm_7": [
            {"date": "2024-10-01", "type": "Preventive", "technician": "A. Weber", "notes": "Joint calibration. Firmware updated to v3.2.1."},
            {"date": "2024-06-18", "type": "Corrective",  "technician": "A. Weber", "notes": "Gripper replaced. Cycle time restored to nominal."},
        ],
    }
    key = machine_id.lower()
    logs = simulated_logs.get(key, [])
    return {
        "machine": machine_id,
        "events": logs[:last_n],
        "count": len(logs[:last_n]),
    }


# ---------------------------------------------------------------------------
# Tool registry — the agent uses this to decide which tool to call
# ---------------------------------------------------------------------------

TOOLS = {
    "search_knowledge_base": {
        "fn": tool_search_knowledge_base,
        "description": "Search technical documents, manuals and reports in the knowledge base.",
        "keywords": ["what", "how", "explain", "describe", "manual", "procedure", "specification"],
    },
    "get_sensor_data": {
        "fn": tool_get_sensor_data,
        "description": "Get live or recent sensor readings for a specific machine.",
        "keywords": ["temperature", "vibration", "cycle", "speed", "pressure", "sensor", "reading", "current"],
    },
    "get_maintenance_log": {
        "fn": tool_get_maintenance_log,
        "description": "Retrieve maintenance history for a machine.",
        "keywords": ["maintenance", "repair", "log", "history", "last service", "replaced", "technician"],
    },
}
