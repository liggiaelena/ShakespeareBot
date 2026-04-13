import chromadb
from config import CHROMA_DB_PATH


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name="agribot_docs",
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas or [{}] * len(ids),
        )

    def query(self, embedding: list[float], top_k: int = 3) -> list[str]:
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )
        return results["documents"][0]

    def count(self) -> int:
        return self.collection.count()
