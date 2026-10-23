# Digital Twin Q&A Agent

A local-first AI agent that lets factory engineers ask questions in natural language about their equipment, maintenance history, and technical documentation - grounded in real factory data, with no data leaving the plant network.

Built as a design and implementation exercise for a smart factory context.

---

## What it does

- Answers questions about factory equipment using a **RAG pipeline** (FAISS + sentence-transformers)
- Retrieves **live sensor data** and **maintenance history** from the digital twin
- Generates answers using a **local LLM via Ollama** (phi3:mini by default)
- Every answer cites its **source document or data stream**
- Runs entirely **on-premise** - no cloud APIs, no data leaves the factory network

## Example questions

- *"What is the maintenance schedule for conveyor A3?"*
- *"What is the current vibration level of robot arm 7?"*
- *"When was robot arm 7 last serviced?"*
- *"What are the safety procedures before working on conveyor A3?"*
- *"Explain the digital twin architecture used in this factory."*

---

## Architecture

```
User query
    │
    ▼
Agent Orchestrator (agent.py)
    ├── Query classifier → selects tools
    ├── Retriever tool → FAISS vector search over document knowledge base
    ├── Sensor data tool → digital twin API (simulated)
    └── Maintenance log tool → factory database (simulated)
    │
    ▼
Local LLM (Ollama / phi3:mini)
    │
    ▼
Answer + source citations
```

See [docs/design.md](docs/design.md) for the full design document including requirements, design decisions, evaluation strategy and deployment plan.

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- phi3:mini pulled: `ollama pull phi3:mini`

### Install

```bash
pip install -r requirements.txt
```

### Build the document index

```bash
python src/main.py
# Index is built automatically on first run
```

### Run

**CLI:**
```bash
python src/main.py
```

**Web UI:**
```bash
streamlit run src/ui.py
```

**Docker:**
```bash
docker build -t dt-agent .
docker run -p 8501:8501 dt-agent
```

Open http://localhost:8501 in your browser.

---

## Project structure

```
├── src/
│   ├── agent.py       # Agent orchestrator + LLM integration
│   ├── retriever.py   # Document chunking, embedding, FAISS index
│   ├── tools.py       # Tool definitions (retrieval, sensor, maintenance)
│   ├── ui.py          # Streamlit web interface
│   └── main.py        # CLI entry point
├── data/
│   └── sample_docs/   # Factory manuals and specs (add your own here)
├── docs/
│   └── design.md      # Full software design document
├── requirements.txt
└── Dockerfile
```

## Adding your own documents

Drop `.txt` or `.md` files into `data/sample_docs/`, then rebuild the index:

```bash
python -c "from src.retriever import Retriever; r = Retriever(); r.build_index()"
```

---

## Tech stack

| Component | Technology | Reason |
|-----------|------------|--------|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Fast, lightweight, multilingual |
| Vector search | FAISS (CPU) | Local, no server needed |
| LLM inference | Ollama + phi3:mini | On-premise, GPU-optional |
| Web UI | Streamlit | Rapid, clean, factory-friendly |
| Packaging | Docker | Consistent deployment |

---

## Author

Harikrishnan Changaramkulath  
M.Sc. Digital Engineering, OVGU Magdeburg  
[github.com/harikrishhnanC98](https://github.com/harikrishhnanC98)
