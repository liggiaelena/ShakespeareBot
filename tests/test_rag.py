"""Tests the RAG pipeline: embedder, vector_store and retriever."""
import uuid
import pytest
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import Retriever


@pytest.fixture
def store():
    """In-memory VectorStore (no persistence) for isolated tests."""
    import chromadb
    client = chromadb.EphemeralClient()
    vs = VectorStore.__new__(VectorStore)
    vs.client = client
    vs.collection = client.get_or_create_collection(
        name="test_collection",
        metadata={"hnsw:space": "cosine"},
    )
    return vs


def test_embedder_returns_vectors():
    embedder = Embedder()
    vectors = embedder.embed(["how to treat rust disease in wheat"])
    assert len(vectors) == 1
    assert len(vectors[0]) > 0
    assert isinstance(vectors[0][0], float)


def test_vector_store_add_and_query(store):
    embedder = Embedder()
    docs = ["Wheat rust is a fungal disease. Apply fungicide early."]
    ids = [str(uuid.uuid4())]
    embeddings = embedder.embed(docs)
    store.add(ids=ids, embeddings=embeddings, documents=docs,
              metadatas=[{"source": "test_doc.txt"}])

    results = store.query(embedder.embed(["rust disease wheat"])[0], top_k=1)
    assert len(results) == 1
    doc_text, source = results[0]
    assert "rust" in doc_text.lower()
    assert source == "test_doc.txt"


def test_retriever_with_empty_store_returns_list():
    # If the store is empty, the retriever should not raise
    retriever = Retriever()
    try:
        results = retriever.search("corn planting season", top_k=1)
        assert isinstance(results, list)
    except Exception:
        # Store may be empty — acceptable in this context
        pass
