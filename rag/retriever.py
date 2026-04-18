from rag.embedder import Embedder
from rag.vector_store import VectorStore
from config import RAG_TOP_K

_RERANK_KEEP = 2  # after fetching RAG_TOP_K candidates, keep the best 2 by word overlap


class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()

    def search(self, query: str, top_k: int = RAG_TOP_K) -> list[str]:
        """Fetches top_k chunks by vector similarity, reranks by query-word overlap,
        returns the top 2 as source-prefixed strings:
        '[Source: pg16966.txt] <chunk text>'
        """
        embedding = self.embedder.embed([query])[0]
        results = self.store.query(embedding, top_k=top_k)  # list[tuple[str, str]]

        # Rerank: count how many distinct query words appear in each chunk
        query_words = set(query.lower().split())
        scored: list[tuple[int, str, str]] = []
        for doc, source in results:
            score = sum(1 for w in query_words if w in doc.lower())
            scored.append((score, doc, source))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            f"[Source: {source}] {doc}"
            for _, doc, source in scored[:_RERANK_KEEP]
        ]
