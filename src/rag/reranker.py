# src/rag/reranker.py - Cross-encoder reranking + MMR diversity

import numpy as np

from src.rag.embedder import Embedder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Second-pass reranking with a cross-encoder.

    Difference vs the embedding model (bi-encoder):
    - Bi-encoder: query and chunk are embedded SEPARATELY, compared
      with a dot product. Fast (can pre-compute all chunks), but
      the model never sees query+chunk TOGETHER.
    - Cross-encoder: query and chunk go through the transformer
      TOGETHER as one input. Slower (must run for every pair), but
      the model sees the exact interaction, so scores are far more
      accurate for ranking.

    This is the same idea as Cohere Rerank / BGE-Reranker, just
    running locally with a small free model.
    """

    def __init__(self, model_name: str = RERANK_MODEL):
        self.model_name = model_name
        self.model = None
        self.embedder = Embedder()

    def _load(self):
        if self.model is None:
            print(
                f"    🎯 Loading reranker: {self.model_name}"
            )
            print(
                "    ⏳ First time downloads ~22MB..."
            )
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            print("    ✅ Reranker loaded!")
        return self.model

    def rerank(
        self,
        query: str,
        candidates: list,
        top_k: int = 3
    ) -> list:
        """
        Re-score (query, chunk_text) pairs, return top_k reordered.

        Each result keeps all its original fields plus:
        - rerank_score: cross-encoder score
        - rerank_rank:  1-based position after reranking
        """
        if not candidates:
            return []

        model = self._load()

        pairs = [(query, c["text"][:4000]) for c in candidates]
        scores = model.predict(pairs)

        for chunk, score in zip(candidates, scores):
            chunk["rerank_score"] = float(score)

        ordered = sorted(
            candidates,
            key=lambda c: c["rerank_score"],
            reverse=True
        )

        for rank, chunk in enumerate(ordered, 1):
            chunk["rerank_rank"] = rank

        return ordered[:top_k]

    def mmr_select(
        self,
        query: str,
        candidates: list,
        top_k: int = 3,
        lambda_param: float = 0.7
    ) -> list:
        """
        Maximal Marginal Relevance - pick results that are BOTH
        relevant to the query AND different from what we already
        picked.

        score(doc) = λ * sim(query, doc)
                    - (1-λ) * max(sim(doc, already_picked))

        λ close to 1  -> pure relevance (no diversity)
        λ close to 0  -> pure diversity

        Fixes the classic RAG failure where all 3 chunks come from
        the same paragraph of the same document.
        """
        if not candidates:
            return []

        query_vec = self.embedder.embed_text(query)
        doc_vecs = self.embedder.embed_texts(
            [c["text"][:1000] for c in candidates]
        )

        query_sim = np.array(query_vec) @ np.array(doc_vecs).T
        doc_sim = np.array(doc_vecs) @ np.array(doc_vecs).T

        remaining = set(range(len(candidates)))
        selected = []
        mmr_scores = {}

        while remaining and len(selected) < top_k:
            best_idx = None
            best_score = -1.0

            for i in remaining:
                if selected:
                    max_selected_sim = max(
                        doc_sim[i][j] for j in selected
                    )
                else:
                    max_selected_sim = 0.0

                score = (
                    lambda_param * query_sim[i]
                    - (1 - lambda_param) * max_selected_sim
                )

                if score > best_score:
                    best_score = score
                    best_idx = i

            selected.append(best_idx)
            mmr_scores[best_idx] = best_score
            remaining.remove(best_idx)

        ordered = []
        for rank, idx in enumerate(selected, 1):
            chunk = dict(candidates[idx])
            chunk["mmr_score"] = round(mmr_scores[idx], 4)
            chunk["mmr_rank"] = rank
            ordered.append(chunk)

        return ordered
