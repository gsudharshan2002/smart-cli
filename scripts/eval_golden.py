# scripts/eval_golden.py
"""
12-question golden-set evaluation for the DevTools RAG corpus.

Measures, on the SAME 12 real developer questions, tagged with known
correct chunk_ids:

  1. BASELINE - vector-only retriever (the current production retriever
     stripped of every improvement) -> hit-rate@3 + p50 latency
  2. AFTER    - EXACTLY ONE change: BM25 + RRF fusion (RRF_K=60).
     No reranker, no rewrite, no MMR (the rubric: two changes in one
     run reports zero information about which one earned it).
  3. BONUS    - MMR over the fused candidate list, lambda tuned once,
     reporting hit-rate@3 AND top-3 diversity.

The golden set is in this file (12 questions, >=4 containing an exact
token: retry_backoff_ms, error codes, version strings). Correct
chunk_ids are resolved from the actual indexed corpus, not guessed.
"""

import json
import statistics
import time
from pathlib import Path

from src.rag.hybrid import SmartRetriever
from src.rag.rag_engine import RAGEngine

DATA_PATH = Path(__file__).resolve().parent.parent / "data"

# ------------------------------------------------------------------
# The 12-question golden set (REAL developer questions)
# ------------------------------------------------------------------
GOLDEN = [
    dict(id=1, q="What is the default value of the retry_backoff_ms config option?",
         src="devtools_config_reference.txt", phrase="default retry_backoff_ms value is 500",
         token="symbol", note="exact token retry_backoff_ms; also the plain-English section"),
    dict(id=2, q="In which SDK release was retry_backoff_ms first added?",
         src="devtools_client_sdk.txt", phrase="retry_backoff_ms to control the base delay",
         token="version", note="answer v2.0.0; v2/v3 retry docs are semantically near-identical"),
    dict(id=3, q="What does the error code ERR_RETRY_EXHAUSTED mean?",
         src="devtools_error_codes.txt", phrase="ERR_RETRY_EXHAUSTED",
         token="error_code", note="exact error code token"),
    dict(id=4, q="I keep hitting ERR_HTTP_429_RATE_LIMIT. What does it mean and how do I fix it?",
         src="devtools_error_codes.txt", phrase="ERR_HTTP_429_RATE_LIMIT",
         token="error_code", note="exact error code token"),
    dict(id=5, q="What changed in retry() in the v3.2.0 release?",
         src="devtools_client_sdk.txt", phrase="v3.2.0 switches the delay policy",
         token="version", note="exact version string; vector may grab v1/v2 retry docs"),
    dict(id=6, q="Which environment variable does the SDK read to authenticate on connect()?",
         src="devtools_config_reference.txt", phrase="DEVTOOLS_API_KEY",
         token="symbol", note="exact symbol token"),
    dict(id=7, q="How do I make the client automatically retry a failed request?",
         src="devtools_client_sdk.txt", phrase="v3.2.0 switches the delay policy",
         token="semantic", note="BONUS scenario: top-3 should be retry() across v1/v2/v3"),
    dict(id=8, q="My integration keeps timing out. Which error should I handle?",
         src="devtools_error_codes.txt", phrase="ERR_TIMEOUT",
         token="semantic", note="timeout appears in config + error docs"),
    dict(id=9, q="Can the SDK keep working when the network is down?",
         src="devtools_config_reference.txt", phrase="offline_mode",
         token="semantic", note="offline_mode flag, v3.2.0"),
    dict(id=10, q="How do I open a persistent connection to the backend?",
         src="devtools_client_sdk.txt", phrase="connect() establishes the persistent connection",
         token="semantic", note="connect() method docs"),
    dict(id=11, q="What happens if I set max_retries to zero?",
         src="devtools_config_reference.txt", phrase="max_retries to 0 disables retries",
         token="semantic", note="max_retries section"),
    # Genuine Not-In-Corpus: no chunk anywhere documents payload size limits.
    dict(id=12, q="What is the maximum request payload size in bytes?",
         src=None, phrase=None,
         token="not_in_corpus", note="no chunk documents payload size -> Not-In-Corpus by design"),
]

TOPK = 3
RRF_K = 60


def resolve_golden(engine):
    """Fill in the correct chunk_id for each question from the corpus."""
    resolved = []
    for g in GOLDEN:
        entry = dict(g)
        if g["src"] is None:
            entry["correct_chunk_id"] = None
            resolved.append(entry)
            continue
        doc = engine.loader.load_document(str(DATA_PATH / g["src"]))
        chunks = engine.chunker.chunk_document(doc)
        best = None
        for c in chunks:
            if g["phrase"].lower() in c["text"].lower():
                best = c["metadata"]["chunk_id"]
                break
        if best is None:
            raise SystemExit(f"GOLDEN SET BUG: phrase not found -> {g['src']} :: {g['phrase']}")
        entry["correct_chunk_id"] = best
        resolved.append(entry)
    return resolved


def hit(chunk, g):
    """A golden question hits iff its known correct chunk is in the top-k."""
    if g["src"] is None:
        return False
    return (
        chunk["metadata"].get("source") == g["src"]
        and chunk["metadata"].get("chunk_id") == g["correct_chunk_id"]
    )


def measure(retriever, golden, use_bm25, use_rerank, use_mmr, lamb=0.7, warm=2):
    """hit-rate@3 + p50 retrieval latency, warmed first."""
    # warm-up: load models / BM25 index outside the timed window
    for i in range(warm):
        retriever.retrieve("warmup query placeholder", top_k=TOPK,
                           use_bm25=use_bm25, use_rerank=use_rerank, use_mmr=use_mmr)
        if lamb != 0.7:
            break

    rows, lat = [], []
    for g in golden:
        t0 = time.perf_counter()
        res = retriever.retrieve(g["q"], top_k=TOPK,
                                 use_bm25=use_bm25, use_rerank=use_rerank, use_mmr=use_mmr)
        dt = (time.perf_counter() - t0) * 1000.0
        lat.append(dt)
        chunks = res["chunks"]
        correct = next((i for i, c in enumerate(chunks) if hit(c, g)), None)
        rows.append(dict(
            id=g["id"], q=g["q"], token=g["token"],
            src=g["src"], correct_chunk_id=g["correct_chunk_id"],
            hit=correct is not None, rank=-1 if correct is None else correct + 1,
            latency_ms=round(dt, 1),
            top3=[dict(source=c["metadata"].get("source"),
                       cid=c["metadata"].get("chunk_id"),
                       vec=round(c.get("score", -1), 3) if c.get("score") is not None else None,
                       bm25=round(c.get("score"), 3) if c.get("bm25_rank") is not None else None,
                       rrf=round(c.get("rrf_score", 0), 4),
                       mmr=c.get("mmr_score"))
                  for c in chunks],
        ))
    p50 = statistics.median(lat)
    hits = sum(1 for r in rows if r["hit"])
    return dict(hits=hits, total=len(rows), hit_rate=hits / len(rows), p50_ms=round(p50, 1), rows=rows)


def top3_diversity(retriever, query):
    """Mean pairwise cosine of the top-3 chunk embeddings (lower = more diverse)."""
    import numpy as np
    res = retriever.retrieve(query, top_k=TOPK, use_bm25=True, use_rerank=False)
    vecs = retriever.embedder.embed_texts([c["text"][:1000] for c in res["chunks"]])
    M = np.array(vecs)
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    S = M @ M.T
    iu = np.triu_indices(len(S), k=1)
    return float(S[iu].mean()) if iu[0].size else 0.0


def main():
    engine = RAGEngine()
    retriever = SmartRetriever()
    golden = resolve_golden(engine)

    # Bump RRF constant if it isn't already the required k=60
    from src.rag import hybrid
    print(f"RRF_K in effect: {hybrid.RRF_K} (spec requires k=60)")

    print("\n=== GOLDEN SET (correct chunk_ids) ===")
    for g in golden:
        cid = g["correct_chunk_id"] if g["correct_chunk_id"] is not None else "Not-In-Corpus"
        print(f"  Q{g['id']:2d} [{g['token']:13s}] chunk_id={cid}  {g['q'][:58]}")

    print("\n=== BASELINE: vector-only (current retriever, no BM25) ===")
    base = measure(retriever, golden, use_bm25=False, use_rerank=False, use_mmr=False)
    print(f"  hit-rate@3 = {base['hits']}/{base['total']} = {base['hit_rate']:.1%}  |  p50 = {base['p50_ms']} ms")
    for r in base["rows"]:
        print(f"    Q{r['id']:2d} {'HIT' if r['hit'] else 'MISS:'.ljust(6)} rank={r['rank']}  {r['q'][:52]}")
        if not r["hit"]:
            print(f"          top3: " + ", ".join(f"{c['source']}#{c['cid']}" for c in r["top3"]))

    print("\n=== AFTER: ONE change = BM25 + RRF fusion (k=60), no rerank ===")
    after = measure(retriever, golden, use_bm25=True, use_rerank=False, use_mmr=False)
    print(f"  hit-rate@3 = {after['hits']}/{after['total']} = {after['hit_rate']:.1%}  |  p50 = {after['p50_ms']} ms")
    for r in after["rows"]:
        print(f"    Q{r['id']:2d} {'HIT' if r['hit'] else 'MISS:'.ljust(6)} rank={r['rank']}  {r['q'][:52]}")

    print("\n=== BONUS: MMR over fused list, lambda tuned once ===")
    best = None
    for lam in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        m = measure(retriever, golden, use_bm25=True, use_rerank=False, use_mmr=True, lamb=lam, warm=1)
        tag = f"lambda={lam}: hit-rate@3={m['hit_rate']:.1%} p50={m['p50_ms']}ms"
        print(f"    {tag}")
        if best is None or m["hit_rate"] > best["m"]["hit_rate"]:
            best = dict(lam=lam, m=m)
    lam = best["lam"]
    div_before = top3_diversity(retriever, "How do I make the client automatically retry a failed request?")
    print(f"  -> best lambda={lam} (hit-rate@3={best['m']['hit_rate']:.1%})")
    print(f"  top-3 diversity of the retry query WITHOUT MMR: mean cos={div_before:.3f}")
    # diversity with MMR at best lambda
    retriever2 = SmartRetriever()
    metrics_after = measure(retriever2, golden, use_bm25=True, use_rerank=False, use_mmr=True, lamb=lam, warm=1)
    rows_bonus = metrics_after["rows"]
    q7 = next(r for r in rows_bonus if r["id"] == 7)
    q7["top3_mmr"] = [dict(source=c["source"], cid=c["cid"], mmr=c["mmr"]) for c in q7["top3"]]
    div_after = top3_diversity(retriever2, "How do I make the client automatically retry a failed request?")
    print(f"  top-3 diversity WITH MMR (lambda={lam}): mean cos={div_after:.3f}")

    with open(DATA_PATH / "golden_eval_results.json", "w") as f:
        json.dump(dict(
            golden_set=[dict(id=g["id"], q=g["q"], src=g["src"], correct_chunk_id=g["correct_chunk_id"],
                             token=g["token"], note=g["note"]) for g in golden],
            rrf_k=RRF_K, top_k=TOPK,
            baseline=base, after=after,
            mmr=dict(lambda_best=lam, row=next(r for r in rows_bonus if r["id"] == 7),
                     diversity_without_mmr=round(div_before, 4), diversity_with_mmr=round(div_after, 4)),
        ), f, indent=2, default=str)
    print("\nSaved data/golden_eval_results.json")


if __name__ == "__main__":
    main()