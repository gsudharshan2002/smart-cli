import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")


EMBEDDING_MODEL = "all-MiniLM-L6-v2"


CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 250      # overlap between chunks

# Hard cap on chunk length in tokens.  A token ≈ 4 chars, so the
# default of 300 tokens ≈ 1 200 chars.  This prevents a single chunk
# from blowing up the retrieval context or the LLM prompt.
# (Uses a simple estimate: tokens ≈ len(text) / 4.)
MAX_CHUNK_TOKENS = 300
MAX_CHUNK_CHARS = MAX_CHUNK_TOKENS * 4  # ≈ 1 200 chars

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