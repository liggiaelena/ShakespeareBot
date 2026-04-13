from rag.embedder import Embedder
from rag.vector_store import VectorStore
from config import RAG_TOP_K


class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()

    def search(self, query: str, top_k: int = RAG_TOP_K) -> list[str]:
        """Returns the most relevant chunks for the query."""
        embedding = self.embedder.embed([query])[0]
        return self.store.query(embedding, top_k=top_k)
