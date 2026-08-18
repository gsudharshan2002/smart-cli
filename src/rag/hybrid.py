# src/rag/hybrid.py - Hybrid retrieval (semantic + keyword) with RRF fusion

from src.rag.bm25 import BM25Index
from src.rag.embedder import Embedder
from src.rag.vectordb import VectorDB
from src.rag.reranker import CrossEncoderReranker
from src.rag.query_rewriter import QueryRewriter
from src.rag.config import TOP_K

RRF_K = 60  # RRF constant (standard value from the paper)


def fuse_rrf(
    ranked_lists: list,
    top_k: int = TOP_K
) -> list:
    """
    Reciprocal Rank Fusion.

    Each ranked list contributes 1 / (RRF_K + rank) to every
    chunk it contains. A chunk that is #1 in BOTH lists beats a
    chunk that is #1 in only one.

    Why ranks, not scores?
    - Vector scores and BM25 scores live on completely different
      scales (cosine similarity ~0.3-1.0 vs BM25 ~0-20)
    - You can't average them meaningfully
    - Ranks are scale-free, so fusion just works
    """
    fused = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, 1):
            chunk_id = chunk["id"]
            score = 1.0 / (RRF_K + rank)

            if chunk_id in fused:
                fused[chunk_id]["rrf_score"] += score
                fused[chunk_id]["lists_hit"] += 1
            else:
                chunk_copy = dict(chunk)
                chunk_copy["rrf_score"] = score
                chunk_copy["lists_hit"] = 1
                chunk_copy["rrf_rank"] = None  # set below
                fused[chunk_id] = chunk_copy

    ordered = sorted(
        fused.values(),
        key=lambda c: c["rrf_score"],
        reverse=True
    )

    for rank, chunk in enumerate(ordered, 1):
        chunk["rrf_rank"] = rank

    return ordered[:top_k]


class SmartRetriever:
    """
    The Week 5 retriever: composes every retrieval technique
    so you can compare them one at a time.

    Pipeline (each stage toggleable):
    Query
      ├─ [rewrite]  LLM rewrites a messy question into a clean query
      ├─ [HyDE]     LLM writes a hypothetical answer, embed THAT instead
      ├─ vector search  (embeddings, semantic meaning)
      ├─ BM25 search    (keyword, exact terms)
      ├─ RRF fusion     (combine the two ranked lists)
      ├─ [rerank]       cross-encoder re-scores the top candidates
      └─ [MMR]          diversity filter on top of rerank
    """

    def __init__(self):
        self.embedder = Embedder()
        self.db = VectorDB(quiet=True)
        self.bm25 = None          # lazy: only when BM25 is turned on
        self.reranker = CrossEncoderReranker()
        self.rewriter = QueryRewriter()

    def _vector_search(self, query: str, top_k: int) -> list:
        """Pure semantic search (same as the Week 4 retriever)."""
        query_embedding = self.embedder.embed_text(query)
        results = self.db.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        # Keep the same 0.3 relevance filter Week 4 uses,
        # so "before" in the eval is exactly the old behavior.
        filtered = [r for r in results if r["score"] > 0.3]

        for rank, chunk in enumerate(filtered, 1):
            chunk["vector_rank"] = rank
            # Preserve original vector score for display
            chunk["vector_score"] = chunk["score"]

        return filtered

    def _bm25_search(self, query: str, top_k: int) -> list:
        """Pure keyword search."""
        if self.bm25 is None:
            self.bm25 = BM25Index()
        return self.bm25.search(query, top_k=top_k)

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        use_bm25: bool = True,
        use_rerank: bool = False,
        use_rewrite: bool = False,
        use_hyde: bool = False,
        use_mmr: bool = False,
        verbose: bool = False
    ) -> dict:
        """
        Full hybrid retrieval with per-stage results for inspection.

        Returns:
        {
            search_query:  the query actually used for search,
            vector_hits:   list of {id, text, metadata, score, vector_rank},
            bm25_hits:     list of {..., score, bm25_rank} or [],
            fused_hits:    list after RRF (with rrf_score, lists_hit),
            chunks:        final top_k list (after rerank/MMR),
            warnings:      notes worth showing the user
        }
        """
        search_query = query
        warnings = []

        # ✅ Step 0: Query rewriting / HyDE (optional)
        if use_rewrite or use_hyde:
            search_query = self.rewriter.rewrite(
                query,
                hypothetically=use_hyde
            )
            warnings.append(
                f"Rewritten search query: '{search_query}'"
            )

        # ✅ Step 1: semantic search
        vector_hits = self._vector_search(
            search_query, top_k=top_k * 2
        )

        # ✅ Step 2: keyword search
        bm25_hits = []
        if use_bm25:
            bm25_hits = self._bm25_search(
                search_query, top_k=top_k * 2
            )

        # ✅ Step 3: RRF fusion
        fused_hits = fuse_rrf(
            [vector_hits, bm25_hits],
            top_k=top_k * 2
        )

        if use_bm25 and fused_hits:
            warnings.append(
                "RRF fused vector + BM25 ranked lists "
                f"({sum(1 for c in fused_hits if c['lists_hit'] > 1)} "
                "chunks matched in both)"
            )

        # ✅ Step 4: rerank (optional)
        if use_rerank and fused_hits:
            fused_hits = self.reranker.rerank(
                query, fused_hits, top_k=top_k * 2
            )
            warnings.append(
                "Cross-encoder reranked the fused candidates"
            )

        # ✅ Step 5: MMR diversity (optional)
        if use_mmr and fused_hits:
            fused_hits = self.reranker.mmr_select(
                search_query, fused_hits, top_k=top_k
            )
            warnings.append("MMR diversity filter applied")

        final = fused_hits[:top_k]

        if verbose:
            print(
                f"    🔍 Search query: '{search_query[:60]}'"
            )
            print(
                f"    ✅ Retrieved {len(final)} chunks "
                f"(vector={len(vector_hits)}, "
                f"bm25={len(bm25_hits)})"
            )

        return {
            "search_query": search_query,
            "vector_hits": vector_hits,
            "bm25_hits": bm25_hits,
            "fused_hits": fused_hits,
            "chunks": final,
            "warnings": warnings
        }

    def build_context(self, chunks: list) -> str:
        """Turn retrieved chunks into the context block for the LLM."""
        from src.rag.retriever import build_context
        return build_context(chunks)
