# Retrieval Lab Guide (Dev Tools for the Merged RAG)

Your task: take a set of failing questions, sort each into
**"wrong document fetched"** vs **"right document, wrong answer"**, make
**one** improvement, and measure **hit-rate@3 before and after** — with a
number, not a feeling.

Everything you need is already built into this project. This guide tells
you what to study, what to run, and how to read the numbers.

---

## 1. The two kinds of failure

| Kind | What happens | Evidence | Where to fix |
|---|---|---|---|
| **A. Wrong document fetched** | The correct policy never reached the LLM | Retrieved chunks are all from the wrong file; answer cites wrong source | Retrieval (BM25, hybrid, rerank) |
| **B. Right document, wrong answer** | The correct chunk WAS retrieved, but the answer is bad | Correct source in top-k, yet answer missed the number / hallucinated | Prompting, chunking, or a stronger model |

> Why not just switch to a smarter model? Because if the app fetched
> the wrong document, a smarter model changes nothing — it reads the
> same wrong text and confidently answers from it. Pay more, fix nothing.

---

## 2. What you built (the files)

| File | What it teaches |
|---|---|
| `src/rag/bm25.py` | Keyword search. Exact terms, codes, numbers embeddings miss |
| `src/rag/hybrid.py` | RRF fusion — combining two ranked lists that use different scales |
| `src/rag/reranker.py` | Cross-encoder reranking + MMR diversity |
| `src/rag/query_rewriter.py` | Query rewriting + HyDE (hypothetical document embeddings) |
| `src/rag/evaluator.py` | hit-rate@k, recall@k, MRR + before/after comparison |
| `data/eval_questions.json` | 18 labeled questions with ground truth |
| `src/features/retrieval_lab.py` | Retrieval Lab — dev tools (RAG menu → option 6) |

> **Merged, not separate.** The main RAG chat (`RAGEngine.query`,
> RAG menu → option 3) now runs the improved pipeline end to end:
> hybrid retrieval (vector + BM25 + RRF) + cross-encoder rerank.
> The toggles live in `src/rag/config.py` (`USE_HYBRID`,
> `USE_RERANK`, `USE_QUERY_REWRITE`). The Retrieval Lab is the dev
> toolbox around that merged pipeline — inspection, measurement,
> and failure labeling for your PR.

---

## 3. The inspection view (the mentor's #1 check)

Run: RAG → **6. Retrieval Lab** → **2. Inspection view**, ask:

> "Can I take a second part-time job while working here?"

You'll see, side by side:

1. **BEFORE** — pure vector search: all 3 chunks from the *Remote Work*
   policy (it mentions "part-time remote working arrangements").
2. **AFTER** — hybrid + rerank: the *Code of Conduct* "Outside
   Employment" section surfaces.
3. **The answer** generated from what was actually fetched.

That one question is your demo case for "wrong document fetched":
embeddings understood *meaning* ("working here" + "part-time") but
missed that the *outside employment* rule is the only place that
answers it. BM25 catches it because the Code of Conduct is the only
document with the words "paid outside work".

---

## 4. The techniques, in learning order

### BM25 — why embeddings miss exact terms
Embeddings are trained on general language; they shine at *meaning* and
fail at *precision*: codes, IDs, dollar amounts, dates. Ask "what is the
max hotel rate abroad?" and embeddings grab any "hotel/travel" chunk;
BM25 scores chunks containing `325`, `international`, `nightly`.

BM25 is smarter word counting:
- **term frequency** — "mileage" appearing 3x in a chunk matters
- **inverse document frequency** — rare words (`skill.md`, `$325`) matter
  more than common ones (`policy`)
- **length normalization** — a huge chunk doesn't win just for being big

### Hybrid search + RRF — why we fuse with ranks
Vector scores (cosine ~0.3–0.99) and BM25 scores (~0–20) live on
different scales. You can't average them. **Reciprocal Rank Fusion**
ignores scores entirely:

```
score(chunk) = 1/(60 + rank_in_vector_list) + 1/(60 + rank_in_bm25_list)
```

Rank 1 in *both* lists beats rank 1 in only one. A chunk BM25 found at
#2 that the vector search missed entirely still scores 1/62.

### Reranking — the second pass
- **Bi-encoder** (your embedder): query and chunk embedded separately,
  compared with a dot product. Fast, pre-computable, but never "sees"
  them together.
- **Cross-encoder**: query + chunk fed into the transformer as one
  input. Slow (runs per pair), but sees exact interaction → better
  ranking. `cross-encoder/ms-marco-MiniLM-L-6-v2` is the free local
  equivalent of Cohere Rerank / BGE-Reranker.

Use bi-encoder to pre-filter ~10 candidates, cross-encoder to re-rank.

### MMR — diversity
All 3 chunks from the same paragraph is a classic failure. MMR penalizes
a candidate that repeats what was already picked:

```
score = λ · relevance(query, doc) − (1−λ) · max similarity(doc, picked)
```

### Query rewriting / HyDE — fix the query first
- **Rewrite**: "whats the max they give back if my hotel is pricey
  while im abroad" → `maximum hotel reimbursement overseas travel policy`
- **HyDE**: instead of rewriting the question, ask the LLM to *write the
  answer as it would appear in the policy*, then embed *that*. Question
  vocabulary and policy vocabulary differ; a fake answer and the real
  paragraph share one.

---

## 5. The numbers (measured on your data, 18 questions, top_k=3)

| Variant | hit-rate@3 | recall@3 | MRR@3 | What it adds |
|---|---|---|---|---|
| `vector` (Week 4 baseline) | **94.44%** (17/18) | 44.44% | 0.944 | nothing |
| `hybrid` (+BM25, RRF) | **100%** (18/18) | 38.89% | 0.972 | keyword search |
| `hybrid+rerank` | **100%** | 50.00% | 0.972 | cross-encoder |
| `rewrite` | 94.44% | 61.11% | 0.944 | LLM query rewrite |
| `hyde` | 94.44% | 44.44% | 0.944 | hypothetical answer |

Three honest lessons hiding in this table — exactly what your mentor
asks about:

1. **BM25 is the one change that fixes the failure** (Q14 "second
   part-time job"). hit-rate@3: 94.44% → 100%. That's your
   before-and-after number.
2. **Not every change helps.** `rewrite` and `hyde` regress hit-rate —
   they change the query *so much* that BM25 no longer matches the
   exact words. Measuring is what reveals this; eyeballing never would.
3. **Recall@3 dips with plain hybrid** (44% → 39%) then recovers with
   rerank (50%). Doc-level hit-rate went up while the exact chunk
   sometimes dropped below rank 3 — reranking pulls it back. Notice
   which failures your change did NOT fix, and why.

---

## 6. How to run the deliverable

The main RAG chat (menu → 13 → 3) already runs the improved
pipeline — that's the merged system you'll PR. The **Retrieval Lab**
(menu → 13 → 6) is where you prove it:

- **3. Before/After** — runs baseline (old vector-only RAG) vs one
  change, prints the metric table + per-question diff + the
  questions your change did NOT fix.
- **5. Label the failures** — for each failing question it shows the
  question, what was fetched, and the generated answer (the evidence),
  then asks you to classify **1** = wrong doc fetched, **2** = right
  doc/wrong answer, **3** = eval label wrong. Saves to
  `data/eval_labels.json`.
- **4. Compare all variants** — exploration: every technique at once.
- **2. Inspection view** — the demo case for your TL: ask "Can I take
  a second part-time job while working here?" and show the wrong
  document fetched by the old RAG vs the right one with the new.

Mentor checklist mapping:

| Mentor asks | Where it lives |
|---|---|
| "Can you show, for a specific failure, which kind it is — with evidence?" | Inspection view (Q14 demo) + labeled failures with fetched chunks + answer |
| "Did you make one change (not five)?" | Before/After menu forces a single variant |
| "Is there a before-and-after number?" | hit-rate@3 table: 94.44% → 100% |
| "Which failures did your change NOT fix?" | The "still failing" list + why rewrite regressed |

---

## 7. Extending it yourself

- **Add questions**: edit `data/eval_questions.json`. Each needs
  `expected_source` (exact filename) and an `expected_phrase` that
  appears inside one chunk (the evaluator finds the exact chunk).
- **Add documents**: drop a PDF/TXT/DOCX into `data/`, re-index
  (RAG → option 2), add eval questions for it.
- **Try a different reranker**: change `RERANK_MODEL` in
  `src/rag/reranker.py` (e.g. `BAAI/bge-reranker-base`).
- **Tune RRF**: change `RRF_K` in `src/rag/hybrid.py` (60 is standard;
  try 30, 100).
- **Tune BM25**: `k1` (term-frequency saturation) and `b` (length
  normalization) in `src/rag/bm25.py`.
- **Label generation failures**: after labeling, for any question
  marked "right document, wrong answer", look at the prompt in
  `RAGEngine.answer_from_chunks` and experiment with it.

## 8. The vocabulary cheat-sheet

- **Retrieval failure** — wrong document fetched
- **Generation failure** — right document, wrong answer
- **BM25** — keyword/lexical search (TF-IDF's smarter cousin)
- **Bi-encoder** — separate embeddings, dot-product compare
- **Cross-encoder** — query+chunk scored together
- **RRF** — rank-based fusion of multiple retrievers
- **MMR** — relevance vs diversity trade-off
- **HyDE** — embed a hypothetical answer instead of the question
- **hit-rate@k** — did the right document appear in top-k?
- **recall@k** — did the exact right chunk appear in top-k?
- **MRR** — how early did the right document appear? (1/rank, averaged)