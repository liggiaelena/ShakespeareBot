import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

WHISPER_MODEL = "small"         # options: tiny, base, small, medium, large
CLAUDE_MODEL = "claude-sonnet-4-6"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # multilingual

CHROMA_DB_PATH = "chroma_db"
DOCS_PATH = "data/docs"

RAG_TOP_K = 2
RAG_CHUNK_SIZE = 300            # words per chunk
RAG_CHUNK_OVERLAP = 50
