import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")


EMBEDDING_MODEL = "all-MiniLM-L6-v2"


CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 250      # overlap between chunks

TOP_K = 3                # retrieve top 3 chunks
COLLECTION_NAME = "smart_cli_docs"

DISTANCE_METRIC = "cosine"


# ⚙️  Retrieval settings for the MERGED RAG pipeline (chat)
# ------------------------------------------------------------------
# These control what the main chat uses. Measured on the eval set
# (18 HR questions, top_k=3): hit-rate@3 vector-only = 94.4%,
# hybrid+rerank = 100%. See src/rag/evaluator.py to re-measure.
#
USE_HYBRID = True        # add BM25 keyword search + RRF fusion
USE_RERANK = True        # cross-encoder second pass on fused hits
USE_QUERY_REWRITE = False  # LLM rewrites messy questions first
                         # (measured to REGRESS this eval set - off)