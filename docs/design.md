# Digital Twin Q&A Agent - Software Design Document

**Author:** Harikrishnan Changaramkulath  
**Date:** October 2026  
**Version:** 1.0

---

## 1. Problem Statement

Modern smart factories generate enormous volumes of data: sensor streams, maintenance logs, CAD specifications, quality reports, and operator manuals. Engineers currently spend significant time manually searching through this information to answer operational questions such as:

- *"Why did conveyor A3 stop last Tuesday?"*
- *"What is the current vibration level of robot arm 7?"*
- *"When is the next scheduled maintenance for the welding station?"*

The goal of this project is to build an AI agent that allows factory engineers and operators to ask questions in plain natural language and receive accurate, source-cited answers - grounded in real factory data and documentation, without hallucination.

---

## 2. Requirements

### 2.1 Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-01 | The system shall accept natural language questions from engineers via a web UI and CLI |
| FR-02 | The system shall retrieve relevant information from a document knowledge base |
| FR-03 | The system shall query live sensor data from the digital twin API |
| FR-04 | The system shall query maintenance history from the factory database |
| FR-05 | The system shall generate a concise, accurate answer using a local LLM |
| FR-06 | Every answer shall cite its source documents or data streams |
| FR-07 | The system shall indicate when it does not have enough information to answer |

### 2.2 Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-01 | **Privacy:** All data stays on-premise. No external API calls for LLM inference. |
| NFR-02 | **Latency:** Response time under 10 seconds for typical queries |
| NFR-03 | **Reliability:** The agent must not fabricate facts. Answers are grounded in retrieved context only. |
| NFR-04 | **Maintainability:** New documents can be added and re-indexed without code changes |
| NFR-05 | **Deployability:** The system runs as a Docker container on existing factory infrastructure |

### 2.3 Constraints

- LLM runs locally via Ollama (no cloud model APIs)
- Vector index stored on local disk (FAISS)
- Agent is read-only - it cannot modify digital twin data or trigger machine actions

---

## 3. System Architecture

### 3.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│              Streamlit Web UI  /  CLI (main.py)                 │
└────────────────────────┬────────────────────────────────────────┘
                         │ query
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Orchestrator                           │
│  1. Classify query → select tools                               │
│  2. Call tools → collect context                                │
│  3. Build prompt with context                                   │
│  4. Call LLM → generate answer                                  │
│  5. Return answer + sources                                     │
└──────┬──────────────────┬───────────────────┬───────────────────┘
       │                  │                   │
       ▼                  ▼                   ▼
┌─────────────┐  ┌──────────────────┐  ┌─────────────────────┐
│  Retriever  │  │  Sensor Data     │  │  Maintenance Log    │
│  (RAG Tool) │  │  Tool            │  │  Tool               │
│             │  │                  │  │                     │
│  FAISS      │  │  Digital Twin    │  │  Factory Database   │
│  Vector     │  │  REST API        │  │  (MES/ERP)          │
│  Index      │  │                  │  │                     │
└──────┬──────┘  └──────────────────┘  └─────────────────────┘
       │
       ▼
┌─────────────┐
│  Document   │
│  Knowledge  │
│  Base       │
│  (.txt,.md) │
└─────────────┘
       │ prompt + context
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Local LLM (Ollama)                           │
│                  phi3:mini / mistral:7b                         │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Key Design Decisions

**Why RAG instead of fine-tuning?**  
Factory documentation changes frequently - new machines are added, procedures are updated, fault codes are revised. A RAG system can be updated by simply adding documents and rebuilding the index. Fine-tuning would require retraining the model for every update, which is impractical in a production environment.

**Chunking strategy**  
Documents are split into 400-character chunks with 80-character overlap. The overlap is intentional: technical specifications often contain measurements or instructions that span sentence boundaries. Without overlap, a split at the wrong point would make a critical fact unretrievable. This strategy was informed by my thesis work on document segmentation, where chunk boundary decisions significantly impacted downstream model performance.

**Why local LLM?**  
Factory data - maintenance logs, sensor readings, quality reports - is sensitive operational data. Sending it to a cloud API (OpenAI, Anthropic, etc.) creates a data privacy and security risk that is unacceptable in an automotive manufacturing context. Running the LLM locally with Ollama keeps all data within the factory network.

**Tool-based architecture over monolithic LLM**  
The agent uses a rule-based query classifier to select which tool(s) to call before invoking the LLM. This is faster and more reliable than having the LLM decide its own tools (ReAct pattern) for a factory setting where predictability matters. The classifier uses keyword matching on the query, making it transparent and debuggable.

**Why Phi-3 Mini?**  
Phi-3 Mini (3.8B parameters) runs efficiently on a single GPU or CPU with 8GB RAM. In evaluation on factual Q&A tasks it performs comparably to much larger models when the context is well-structured - which it is in our case because we supply it with precise retrieved chunks. Larger models (Mistral-7B, LLaMA3) can be swapped in by changing one config line.

---

## 4. Development Plan

| Phase | Scope | Deliverable |
|-------|-------|-------------|
| 1 | Document ingestion, chunking, FAISS index | `retriever.py`, indexed knowledge base |
| 2 | Tool layer (sensor data, maintenance log) | `tools.py` with simulated data sources |
| 3 | Agent orchestrator + LLM integration | `agent.py`, working CLI |
| 4 | Web UI + Docker packaging | `ui.py`, `Dockerfile` |
| 5 | Integration with live digital twin API | Replace simulated data with real API calls |

---

## 5. Evaluation and Testing

### 5.1 Retrieval quality
- **Precision@K**: of the top-K chunks retrieved, what fraction are relevant to the query?
- **Recall@K**: does the retrieved set contain the chunk needed to answer the query?
- Target: Precision@5 > 0.7, Recall@5 > 0.8 on a held-out set of 50 Q&A pairs

### 5.2 Answer quality
- **Faithfulness**: does the answer contain only facts present in the retrieved context?
  Evaluated by checking each claim in the answer against the source chunks.
- **Answer relevance**: does the answer address the question asked?
- These match the evaluation methodology from my master's thesis (WindowDiff, Pk metrics for segmentation; precision/recall for retrieval).

### 5.3 Edge cases
- Query has no relevant documents → agent says "I don't have enough information"
- Live sensor data unavailable → agent falls back to last known value from knowledge base
- Ambiguous machine ID in query → agent lists possible matches and asks for clarification

---

## 6. Deployment

```bash
# Build and run with Docker
docker build -t dt-agent .
docker run -p 8501:8501 dt-agent

# Or run locally (requires Ollama running on localhost:11434)
pip install -r requirements.txt
python src/main.py        # CLI
streamlit run src/ui.py   # Web UI
```

The container exposes port 8501 (Streamlit). In production it would sit behind the factory's internal reverse proxy, accessible only on the plant network.

---

## 7. Future Extensions

- **ReAct agent loop**: let the LLM decide which tools to call dynamically, enabling multi-step reasoning
- **Live digital twin API integration**: replace simulated sensor/maintenance data with real API calls
- **Multilingual support**: factory engineers in German and English - the embedding model already supports both
- **Feedback loop**: engineers rate answers; poor ratings trigger re-indexing or prompt refinement
- **Voice interface**: operators on the shop floor could query hands-free via speech-to-text
