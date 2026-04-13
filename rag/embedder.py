from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()
