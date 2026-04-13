"""
Runs once to populate ChromaDB with Shakespeare PDF documents.
Usage: python -m rag.ingestor
"""
import uuid
from pathlib import Path

import pypdf

from config import DOCS_PATH, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP
from rag.embedder import Embedder
from rag.vector_store import VectorStore


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.strip()) > 30:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def ingest(docs_path: str = DOCS_PATH) -> None:
    embedder = Embedder()
    store = VectorStore()
    path = Path(docs_path)

    docs = list(path.glob("*.pdf")) + list(path.glob("*.txt"))
    if not docs:
        print(f"No PDF or TXT files found in {docs_path}/")
        return

    for doc_file in docs:
        print(f"Ingesting {doc_file.name}...")
        if doc_file.suffix == ".pdf":
            reader = pypdf.PdfReader(str(doc_file))
            full_text = " ".join(
                page.extract_text() or "" for page in reader.pages
            )
        else:
            full_text = doc_file.read_text(encoding="utf-8", errors="ignore")
        chunks = _chunk_text(full_text, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP)

        ids = [str(uuid.uuid4()) for _ in chunks]
        embeddings = embedder.embed(chunks)
        metadatas = [{"source": doc_file.name}] * len(chunks)

        store.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        print(f"  → {len(chunks)} chunks added")

    print(f"\nTotal documents in database: {store.count()}")


if __name__ == "__main__":
    ingest()
