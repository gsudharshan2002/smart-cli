# RAG System Guide

## Overview

The Smart CLI RAG system is a **garbage-collected, modular, production-grade** pipeline that answers questions using your indexed documents. It combines dense vector embeddings with sparse BM25 keyword search (RRF fusion) and cross-encoder reranking to retrieve the *right* document and generate accurate answers.

**Key Achievement**: 94.44% → 100% hit-rate@3 on 18 HR policy questions.

---

## Config Layer (`src/rag/config.py`)

**Purpose**: Central switchboard for the merged pipeline. Toggle features on/off to experiment with retrieval speed vs accuracy.

| Setting | Default | Effect |
|---------|---------|--------|
| `TOP_K = 3` | How many chunks to retrieve |
| `COLLECTION_NAME = "smart_cli_docs"` | ChromaDB collection name |
| `DISTANCE_METRIC = "cosine"` | Similarity measure |
| `USE_HYBRID = True` | Enable BM25 keyword search + RRF fusion |
| `USE_RERANK = True` | Run cross-encoder second pass |
| `USE_QUERY_REWRITE = False` | LLM rewrites queries first (regresses on eval) |

---

## Pipeline Modules

### 1. Document Loader (`src/rag/loader.py`)

**Purpose**: Ingest diverse document formats into plain text.

- `DocumentLoader.load_dir()` → `{path: text}` for all pdfs/docxs in a dir
- `DocumentLoader.load_single(filepath)` → single document
- Supports: PDF (pypdf), DOCX, TXT, Markdown
- Metadata: `{source, page_number, text}`

### 2. Chunker (`src/rag/chunker.py`)

**Purpose**: Break long documents into overlapping chunks (the retrieval granularity unit).

- `TextChunker.chunk(text, source, chunk_size=1000, overlap=200)`
- Output: list of `{text, metadata}` where metadata includes source, chunk_id, page, start_char
- Overlap ensures context isn't cut mid-sentence

### 3. Embedder (`src/rag/embedder.py`)

**Purpose**: Convert text chunks into dense vector embeddings for semantic search.

- Model: `sentence-transformers/all-MiniLM-L6-v2` (small, 384-dim, fast inference)
- Free tier: runs entirely on CPU via `sentence-transformers`
- `Embedder.embed(texts)` → list/array of vectors
- Vectors cached in ChromaDB; regenerate only if `EMBED_MODEL` changes

### 4. Vector Database (`src/rag/vectordb.py`)

**Purpose**: Store, search, update, and commit chunk embeddings locally.

- Wrapper over Chromadb
- `VectorDB.upsert(chunks)` — add/update with deduplication by `source + chunk_id`
- `VectorDB.query(vector, k)` — nearest-neighbor search, returns list of `{id, score, metadata, text}`
- `VectorDB.persist()` — write to disk (called at EOF of `load` and `index`)
- `VectorDB.count()` — total indexed chunks

### 5. BM25 (`src/rag/bm25.py`)

**Purpose**: Sparse keyword retrieval — catches exact terms embeddings miss:
- Dollar amounts (`$325` often broken as tokens)
- Policy codes (`REQ-2024-01`)
- Dates (`2024-06-30`)

- `build_bm25(chunks)` → stores inverted index `{term: [(doc_id, tf), ...]}`
- `bm25.search(query, k)` → top-k docs with `bm25_score`
- No neural model needed — pure statistics, extremely fast

### 6. Hybrid Retriever (`src/rag/hybrid.py`)

**Purpose**: Combine dense + sparse results into a single ranked list.

- Uses **Reciprocal Rank Fusion (RRF)** — ranks lower is better, converts to unified score `1 / (k + rank)`
- `SmartRetriever.retrieve(query, top_k=3, use_bm25=True, use_rerank=True)`:
  1. Retrieve `3*top_k` candidates from each method (vector, BM25)
  2. Merge via RRF → unified ranking
  3. Optionally cross-encoder rerank top 6
- Output: `chunks` with `rrf_score`, `vector_rank`, `bm25_rank`, `rerank_score`, `rerank_rank`, `lists_hit`

### 7. Reranker (`src/rag/reranker.py`)

**Purpose**: Cross-encoder re-scoring — the query and document passage interact.

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (MonoT5-style)
- Takes `(query, passage)` pairs → single relevance score
- `rerank(query, candidates)` returns same list, reordered
- Pros: captures nuanced relevance (even if embedding missed it)
- Cons: slower — only run on top fused results

### 8. Retriever Utilities (`src/rag/retriever.py`)

**Purpose**: Shared helper functions used by the pipeline.

- `build_context(chunks)` → formatted string for LLM prompt (<token limit)
- `label_failure(...)` — for the Retrieval Lab's failure labeling tool

### 9. Evaluator (`src/rag/evaluator.py`)

**Purpose**: Measure and prove the pipeline works.

- Hits 18 labeled HR policy questions with ground truth sources
- Metrics: hit-rate@k, recall@k, MRR@k (Mean Reciprocal Rank)
- `Evaluator.compare(baseline, candidate, top_k)` → `{before, after, delta}`
- Provides evidence for PRs (which questions improved/worsened)

### 10. RAG Engine (`src/rag/rag_engine.py`)

**Purpose**: End-to-end pipeline controller.

```
Index Flow:
  load PDFs/DOCXs → chunk → embed → chroma

Query Flow:
  user question → SmartRetriever (hybrid + BM25 + RRF + rerank)
               → build prompt with top chunks
               → Groq LLM (llama-3.3-70b) → final answer
```

Key insight: retrieval mode printed: `hybrid+rerank`. Configurable via toggle.

---

## Feature Layer

### `src/features/rag_chat.py`

**Purpose**: Interactive CLI chat with documents.

- Menu hierarchy: Main -> RAG -> (1:Index, 2:View DB, 3:Chat, 4:Rebuild, 5:Reset, 6:Lab)
- Chat shows step-by-step pipeline progress (Retrieval, Prompt)
- After answer: optionally dump chunk details with RRF + vector + rerank scores

### `src/features/retrieval_lab.py`

**Purpose**: Retrieval Lab - evaluation and inspection tools for the merged RAG pipeline.

- **Concepts**: why retrieval fails (wrong doc vs wrong answer), each technique explained
- **Inspection**: see what was fetched for a question, side-by-side chunks
- **Before/After**: compare baseline (vector-only) vs merged (hybrid+rerank) with the 18 HR eval questions
- **Compare all variants**: toggle every combo (BM25 off/on, rerank off/on, etc.)
- **Label failures**: manually classify which technique missed the right document
- **Golden set**: 12-question developer eval with hit-rate@3 measurement (see `results.md`)

---

## Data Layer

### `data/`

| File | Purpose |
|------|---------|
| `eval_questions.json` | 18 HR questions with ground truth `source` |
| `eval_labels.json` | Failure classifications for PR evidence |
| Original PDFs | HR policies used for indexing |

---

## Answer Generation (`src/ai_client.py`, `main.py`)

Pipeline uses Groq's `llama-3.3-70b` via streaming:
- System prompt: "You are a helpful policy expert. Only cite the provided context."
- User prompt: `context` (up to 6000 chars) + `question`
- Response streamed back; chunks cited with `[chunk_id: N]`

---

## Development Workflow

1. **Document Ingestion**:
   ```python
   engine = RAGEngine()
   engine.index("path/to/docs")  # load → chunk → embed → store
   ```
   (or main menu option 13 → 1)

2. **Ask Questions**:
   ```python
   result = engine.query("Can I take a second part-time job?")
   print(result["answer"])
   print(result["sources"])
   ```
   (menu option 13 → 3)

3. **Measure Improvement**:
   ```python
   ev = Evaluator()
   ev.compare("vector", "hybrid+rerank", top_k=3)
   ```
   (menu 13 → 6 option 3)

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    RAGEngine (rag_engine.py)               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Document  │  │Chunker   │  │Embedder  │  │Vectordb  │   │
│  │Loader    │  │(1000/200)│  │MiniLM    │  │Chromadb  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │             │             │          │
│       └─────────────────────────────┘             │          │
│                                                 │          │
│                    ┌─────────────────────┐      │          │
│                    │   SmartRetriever    │◄─────┘          │
│                    │  (hybrid.py)        │                 │
│                    │  1. vector search   │                 │
│                    │  2. BM25 keyword    │                 │
│                    │  3. RRF fuse        │                 │
│                    │  4. rerank          │                 │
│                    └─────────┬───────────┘                 │
│                              │                             │
│                              ▼                             │
│                    ┌─────────────────────┐                 │
│                    │ build_context()     │                 │
│                    └─────────┬───────────┘                 │
│                              │                             │
│                              ▼                             │
│                    ┌─────────────────────┐                 │
│                    │ ask_ai() → LlamA 70B│                 │
│                    └─────────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Production Considerations

1. **Relevance Gate** (future): Reject off-topic questions before LLM call
   - Vector score < 0.35 OR rerank < -4 → "I couldn't find anything relevant"
   - Saves cost and prevents hallucination

2. **Cache Invalidation**: Chunker/Embedder detect when `EMBED_MODEL` changes; force rebuild

3. **Security**: ChromaDB runs locally; no PII leaves your machine

4. **Scalability**: 
   - For >1000 docs, consider sharded ChromaDB or Qdrant
   - RRF threshold tuning may be needed per-domain