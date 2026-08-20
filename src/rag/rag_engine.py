# src/rag/rag_engine.py - Full RAG Pipeline

import time
import uuid
from src.rag.loader import DocumentLoader
from src.rag.chunker import TextChunker
from src.rag.embedder import Embedder
from src.rag.vectordb import VectorDB
from src.rag.hybrid import SmartRetriever
from src.rag.retriever import build_context
from src.rag.config import (
    USE_HYBRID,
    USE_RERANK,
    USE_QUERY_REWRITE
)
from src.ai_client import (
    ask_ai,
    ask_rag_structured,
    RAGAnswer,
    RAGGenerationError
)
from rich.console import Console

# Colored Request ID console
console = Console()


class RAGEngine:
    """
    Complete RAG Pipeline

    Index Flow:
    PDF → Load → Chunk → Embed → Store in ChromaDB

    Query Flow:
    Question → Hybrid retrieve (vector + BM25 + RRF + rerank)
             → LLM → Answer
    """

    def __init__(self):
        self.loader = DocumentLoader()
        self.chunker = TextChunker()
        self.embedder = Embedder()
        self.db = VectorDB()
        self.retriever = SmartRetriever()


    def index_document(self, path: str) -> dict:
        """
        Index a single document into ChromaDB

        Steps:
        1. Load document
        2. Chunk text
        3. Create embeddings
        4. Store in ChromaDB
        """

        print(f"\n📥 Indexing document: {path}")

        # ✅ Check if already indexed
        from pathlib import Path
        source = Path(path).name

        if self.db.document_exists(source):
            print(f"    ⚠️  Already indexed: {source}")
            print(f"    ℹ️  Skipping (use re-index to force)")
            return {
                "status": "skipped",
                "source": source,
                "reason": "already indexed"
            }

        # ✅ Step 1: Load
        print("    Step 1: Loading document...")
        doc = self.loader.load_document(path)

        if not doc["text"]:
            return {
                "status": "error",
                "source": source,
                "reason": "no text extracted"
            }

        print(
            f"    ✅ Loaded: {doc['metadata']['words']} words "
            f"| {doc['metadata']['chars']} chars"
        )

        # ✅ Step 2: Chunk
        print("    Step 2: Chunking text...")
        chunks = self.chunker.chunk_document(doc)

        if not chunks:
            return {
                "status": "error",
                "source": source,
                "reason": "no chunks created"
            }

        # ✅ Step 3: Embed
        print("    Step 3: Creating embeddings...")
        chunks_with_embeddings = self.embedder.embed_chunks(
            chunks
        )

        # ✅ Step 4: Store
        print("    Step 4: Storing in ChromaDB...")
        self.db.add_chunks(chunks_with_embeddings)

        print(f"\n    🎉 Successfully indexed: {source}")
        print(f"    📊 {len(chunks)} chunks stored")

        return {
            "status": "success",
            "source": source,
            "chunks": len(chunks),
            "words": doc["metadata"]["words"]
        }

    def index_all_documents(self) -> list:
        """Index all documents in data/ folder"""

        docs = self.loader.list_documents()

        if not docs:
            print("    ❌ No documents found in data/ folder!")
            print("    💡 Add PDF/TXT/DOCX files to data/")
            return []

        print(f"\n📚 Found {len(docs)} documents to index:")
        for d in docs:
            print(f"  📄 {d['name']} ({d['size']})")

        results = []
        for doc in docs:
            result = self.index_document(doc["path"])
            results.append(result)

        # ✅ Summary
        success = sum(
            1 for r in results
            if r["status"] == "success"
        )
        skipped = sum(
            1 for r in results
            if r["status"] == "skipped"
        )
        errors = sum(
            1 for r in results
            if r["status"] == "error"
        )

        print(f"\n📊 Indexing Summary:")
        print(f"  ✅ Success: {success}")
        print(f"  ⏭️  Skipped: {skipped}")
        print(f"  ❌ Errors:  {errors}")

        return results

    # ✅ QUERY PIPELINE

    def query(
        self,
        question: str,
        top_k: int = 3
    ) -> dict:
        """
        Full RAG query pipeline

        Steps:
        1. Retrieve relevant chunks
        2. Build context
        3. Generate answer with LLM
        4. Return answer + sources

        Each query is tracked with a unique Request ID and
        per-stage latency timings in the trace field.
        """

        request_id = str(uuid.uuid4())[:8]
        console.print(
            f"\n[bold cyan]Request ID:[/bold cyan] [cyan]{request_id}[/cyan]"
        )

        # Stage timing
        query_start = time.time()
        trace = {
            "request_id": request_id,
            "timestamp": time.time(),
            "stages": {},
            "total_duration_s": 0.0,
            "total_model_calls": 0,
            "total_cost_usd": 0.0,
        }

        # ✅ Step 1: Retrieve (merged pipeline - hybrid + rerank)
        stage_start = time.time()
        print("    Step 1: Retrieving relevant chunks...")

        retrieval = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            use_bm25=USE_HYBRID,
            use_rerank=USE_RERANK,
            use_rewrite=USE_QUERY_REWRITE,
            verbose=True
        )

        trace["stages"]["retrieval"] = {
            "duration_s": round(time.time() - stage_start, 3),
        }

        chunks = retrieval["chunks"]

        if not chunks:
            trace["stages"]["llm"] = {
                "duration_s": None,
                "details": "No chunks retrieved",
            }
            trace["total_duration_s"] = round(time.time() - query_start, 3)
            trace["total_model_calls"] = 0
            # Display latency (stored in seconds → show as ms)
            ret_ms = round((time.time() - query_start) * 1000, 1)
            console.print(f"    Retrieval: {ret_ms} ms")
            console.print("    LLM: skipped")
            return {
                "question": question,
                "answer": (
                    "I could not find relevant information "
                    "in the documents to answer your question. "
                    "Please make sure documents are indexed "
                    "and try rephrasing your question."
                ),
                "sources": [],
                "chunks_used": 0,
                "context": "",
                "trace": trace,
            }

        mode = "+".join(filter(None, [
            "vector" if not USE_HYBRID else "hybrid",
            "rerank" if USE_RERANK else "",
            "rewrite" if USE_QUERY_REWRITE else "",
        ]))
        print(f"    ℹ️  Retrieval mode: {mode}")

        time.sleep(0.4)

        # ✅ Step 2: Build prompt with context
        print("    Step 2: Building RAG prompt...")
        stage_start = time.time()

        answer = self.answer_from_chunks(question, chunks)

        sources = list(set(
            c["metadata"].get("source", "unknown")
            for c in chunks
        ))

        trace["stages"]["llm"] = {
            "duration_s": round(time.time() - stage_start, 3),
            "details": "answer generated",
        }
        trace["total_duration_s"] = round(time.time() - query_start, 3)
        trace["total_model_calls"] = 1

        # Display latency (stored in seconds → show as ms)
        ret_ms = trace['stages']['retrieval']['duration_s'] * 1000
        llm_ms = trace['stages']['llm']['duration_s'] * 1000
        console.print(f"    Retrieval: {ret_ms:.1f} ms")
        console.print(f"    LLM: {llm_ms:.1f} ms")

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks_used": len(chunks),
            "chunks": chunks,
            "context": build_context(chunks),
            "retrieval": retrieval,
            "trace": trace,
        }

    def answer_from_chunks(
        self,
        question: str,
        chunks: list,
        verbose: bool = False
    ) -> str:
        """
        Generate an LLM answer from a given list of chunks.

        Reused by the Week 5 inspection view and failure labeling,
        so the answer can be judged against exactly what retrieval
        returned (without running the whole pipeline again).

        Security safeguards:
        - Context is wrapped in <retrieved_docs> tags (see
          build_context) so the LLM can identify it as untrusted.
        - The system prompt explicitly warns the LLM to IGNORE any
          instructions found inside retrieved content.
        - This call uses ask_rag_structured which has NO tool/action
          permissions — it can only produce text, never execute
          actions even if injection succeeds.
        - Output is a validated Pydantic model (RAGAnswer) so
          malformed/injected responses are caught.
        """
        context = build_context(chunks)

        if verbose:
            print("    Step 2: Building RAG prompt...")

        rag_prompt = f"""
You are a helpful assistant that answers questions
based ONLY on the provided document context.

{context}

USER QUESTION:
{question}

INSTRUCTIONS:
- Answer ONLY based on the context above
- If the answer is not in the context, say so clearly
- Quote relevant parts when helpful
- Be specific and accurate
- Keep the answer concise (aim for under 250 words)
- After every factual claim, cite the chunk_id it came from, in the exact
  form [chunk_id: X] (X is the chunk_id number shown in that context block above)
- If a claim draws on multiple chunks, cite all of them, e.g. [chunk_id: 3][chunk_id: 7]
- Do not make up information, and never invent a chunk_id that wasn't shown to you
- Return your response as structured JSON matching the requested schema
"""

        if verbose:
            print("    Step 3: Generating answer with LLM...")

        try:
            result = ask_rag_structured(
                prompt=rag_prompt,
                temperature=0.1,
            )
        except RAGGenerationError as e:
            console.print(f"[bold red]⚠️  {e}[/bold red]")
            return (
                "⚠️ Something went wrong while generating your answer. "
                "Please try again in a moment."
            )

        return result.answer

    def reindex_document(self, path: str) -> dict:
        """Force re-index a document"""
        from pathlib import Path
        source = Path(path).name

        print(f"🔄 Re-indexing: {source}")

        # ✅ Delete existing chunks
        collection = self.db.get_collection()
        existing = collection.get(
            where={"source": source}
        )

        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            print(
                f"    🗑️  Deleted {len(existing['ids'])} "
                f"old chunks"
            )

        # ✅ Re-index
        return self.index_document(path)

    def get_stats(self) -> dict:
        """Get full RAG stats"""
        db_stats = self.db.get_stats()
        docs = self.loader.list_documents()

        return {
            "total_chunks": db_stats["total_chunks"],
            "documents_in_folder": len(docs),
            "documents": [
                {
                    "name": d["name"],
                    "size": d["size"],
                    "indexed": self.db.document_exists(
                        d["name"]
                    )
                }
                for d in docs
            ]
        }