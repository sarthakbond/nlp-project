from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory
DATA_DIR = PROJECT_ROOT / "data"

# Sample PDF
PDF_PATH = DATA_DIR / "task.pdf"

# RAG settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"