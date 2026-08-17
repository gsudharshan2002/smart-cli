# src/rag/bm25.py - Keyword search (BM25)

import re

from rank_bm25 import BM25Okapi

from src.rag.vectordb import VectorDB

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    """
    Split text into lowercase word tokens.

    'ERR-4032' -> ['err', '4032']
    'IRS mileage rate' -> ['irs', 'mileage', 'rate']
    """
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    """
    Keyword (lexical) search over the indexed chunks.

    Why BM25 instead of plain word counting?
    - Term frequency: words that appear often in a chunk matter more
    - Document frequency: rare words count more than common words
      (like 'the' is nearly useless, 'ERR-4032' is gold)
    - Length normalization: a 5000-word chunk isn't 10x better just
      because it has more words

    This is exactly what embeddings are BAD at:
    - exact codes, IDs, numbers ('$325', '10 days', 'HR-POL-014')
    - rare technical terms the model never saw in training
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.chunks = []          # parallel to bm25.corpus
        self.lookup = {}          # chunk_id -> index in self.chunks
        self._build()

    def _build(self):
        """Load all chunks from ChromaDB and build the BM25 index."""
        print("    🔤 Building BM25 keyword index over all chunks...")

        db = VectorDB(quiet=True)
        collection = db.get_collection()
        data = collection.get(
            include=["documents", "metadatas"]
        )

        self.chunks = []
        tokenized = []

        for i in range(len(data["ids"])):
            chunk = {
                "id": data["ids"][i],
                "text": data["documents"][i],
                "metadata": data["metadatas"][i]
            }
            self.lookup[chunk["id"]] = i
            self.chunks.append(chunk)
            tokenized.append(tokenize(chunk["text"]))

        if not tokenized:
            print("    ⚠️  No chunks found to index!")
            return

        self.bm25 = BM25Okapi(
            corpus=tokenized,
            k1=self.k1,
            b=self.b
        )

        print(
            f"    ✅ BM25 index ready: {len(tokenized)} chunks"
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0
    ) -> list:
        """
        Score every chunk against the query, return best top_k.

        Each result:
        {
            id, text, metadata,
            score:       BM25 score (higher = better, unbounded)
            bm25_rank:   1-based position in this list
        }
        """
        if self.bm25 is None:
            return []

        tokens = tokenize(query)

        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []
        for rank, idx in enumerate(ranked, 1):
            if scores[idx] <= min_score:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            chunk["bm25_rank"] = rank
            results.append(chunk)

            if len(results) >= top_k:
                break

        return results
