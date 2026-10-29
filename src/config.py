from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory
DATA_DIR = PROJECT_ROOT / "data"

# Sample PDF
PDF_PATH = DATA_DIR / "task.pdf"

# RAG & Chunking settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Topic #9: TF-IDF + Cosine Similarity settings (Official Syllabus Rubric)
DEFAULT_ALGORITHM = "tfidf"  # Options: "tfidf", "dense"
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TOP_K_CHUNKS = 3

# Modern Dense Embedding Model (Optional / Comparison mode)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
