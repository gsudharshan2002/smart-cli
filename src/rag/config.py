import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")


EMBEDDING_MODEL = "all-MiniLM-L6-v2"


CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 200      # overlap between chunks

TOP_K = 3                # retrieve top 3 chunks
COLLECTION_NAME = "smart_cli_docs"

DISTANCE_METRIC = "cosine"