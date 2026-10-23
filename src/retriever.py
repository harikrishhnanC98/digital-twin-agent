"""
retriever.py
------------
Handles document ingestion, chunking, embedding and vector search.
Uses FAISS for local vector storage and sentence-transformers for embeddings.
No external API calls — runs fully on-premise.
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBED_MODEL = "all-MiniLM-L6-v2"   # fast, lightweight, runs on CPU or GPU
CHUNK_SIZE = 400                    # characters per chunk
CHUNK_OVERLAP = 80                  # overlap to preserve context across chunks
INDEX_PATH = "data/index.faiss"
META_PATH = "data/index_meta.pkl"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[dict]:
    """
    Split text into overlapping chunks.

    Overlap ensures that a sentence split across a boundary is still
    retrievable — critical for technical factory documents where a
    single measurement or specification may span multiple lines.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "text": chunk,
                "source": source,
                "start_char": start,
            })
        start += chunk_size - overlap
    return chunks


def load_documents(data_dir: str = "data/sample_docs") -> List[dict]:
    """
    Load all .txt and .md files from the data directory.
    In a production deployment this would connect to the digital twin
    document store (CAD specs, maintenance logs, sensor reports).
    """
    docs = []
    for path in Path(data_dir).glob("**/*"):
        if path.suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            docs.append({"text": text, "source": path.name})
    return docs


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self, embed_model: str = EMBED_MODEL):
        print(f"[Retriever] Loading embedding model: {embed_model}")
        self.model = SentenceTransformer(embed_model)
        self.index: faiss.IndexFlatL2 | None = None
        self.meta: List[dict] = []

    def build_index(self, data_dir: str = "data/sample_docs"):
        """
        Ingest all documents, chunk them, embed each chunk, and store in FAISS.
        Call once (or whenever documents are updated).
        """
        docs = load_documents(data_dir)
        if not docs:
            raise ValueError(f"No documents found in {data_dir}")

        all_chunks: List[dict] = []
        for doc in docs:
            chunks = chunk_text(doc["text"], source=doc["source"])
            all_chunks.extend(chunks)

        print(f"[Retriever] Indexing {len(all_chunks)} chunks from {len(docs)} documents...")
        texts = [c["text"] for c in all_chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype(np.float32))
        self.meta = all_chunks

        # Persist
        os.makedirs("data", exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.meta, f)
        print(f"[Retriever] Index saved ({len(all_chunks)} chunks).")

    def load_index(self):
        """Load a previously built index from disk."""
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError("Index not found. Run build_index() first.")
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            self.meta = pickle.load(f)
        print(f"[Retriever] Index loaded ({len(self.meta)} chunks).")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Embed the query and return the top-k most similar chunks.
        Returns list of (chunk_text, source, distance).
        """
        if self.index is None:
            self.load_index()

        q_vec = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        distances, indices = self.index.search(q_vec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.meta):
                chunk = self.meta[idx]
                results.append((chunk["text"], chunk["source"], float(dist)))
        return results


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    r = Retriever()
    r.build_index()
    hits = r.retrieve("What is the maintenance schedule for conveyor belt A3?")
    for text, src, dist in hits:
        print(f"\n[{src}] (dist={dist:.2f})\n{text[:200]}")
