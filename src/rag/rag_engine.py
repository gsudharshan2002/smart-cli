# src/rag/rag_engine.py - Full RAG Pipeline

import time
from src.rag.loader import DocumentLoader
from src.rag.chunker import TextChunker
from src.rag.embedder import Embedder
from src.rag.vectordb import VectorDB
from src.rag.retriever import Retriever
from src.ai_client import ask_ai


class RAGEngine:
    """
    Complete RAG Pipeline

    Index Flow:
    PDF → Load → Chunk → Embed → Store in ChromaDB

    Query Flow:
    Question → Embed → Retrieve → LLM → Answer
    """

    def __init__(self):
        self.loader = DocumentLoader()
        self.chunker = TextChunker()
        self.embedder = Embedder()
        self.db = VectorDB()
        self.retriever = Retriever()


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
        """

        print(f"\n🔍 Processing query...")

        # ✅ Step 1: Retrieve
        print("    Step 1: Retrieving relevant chunks...")
        retrieval = self.retriever.retrieve_with_context(
            query=question,
            top_k=top_k
        )

        if not retrieval["found"]:
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
                "context": ""
            }

        time.sleep(0.4)

        # ✅ Step 2: Build prompt with context
        print("    Step 2: Building RAG prompt...")

        rag_prompt = f"""
You are a helpful assistant that answers questions
based ONLY on the provided document context.

CONTEXT FROM DOCUMENTS:
{retrieval['context']}

USER QUESTION:
{question}

INSTRUCTIONS:
- Answer ONLY based on the context above
- If the answer is not in the context, say so clearly
- Quote relevant parts when helpful
- Be specific and accurate
- After every factual claim, cite the chunk_id it came from, in the exact
  form [chunk_id: X] (X is the chunk_id number shown in that context block above)
- If a claim draws on multiple chunks, cite all of them, e.g. [chunk_id: 3][chunk_id: 7]
- Do not make up information, and never invent a chunk_id that wasn't shown to you
"""

        time.sleep(0.4)

        # ✅ Step 3: Generate answer
        print("    Step 3: Generating answer with LLM...")

        answer = ask_ai(
            prompt=rag_prompt,
            system=(
                "You are a precise document assistant. "
                "Only answer from provided context. "
                "Never hallucinate or make up information."
            ),
            temperature=0.1
        )

        return {
            "question": question,
            "answer": answer,
            "sources": retrieval["sources"],
            "chunks_used": len(retrieval["chunks"]),
            "chunks": retrieval["chunks"],
            "context": retrieval["context"]
        }

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