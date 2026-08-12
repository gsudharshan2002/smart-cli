from src.rag.embedder import Embedder
from src.rag.vectordb import VectorDB
from src.rag.config import TOP_K


class Retriever:
    """
    Retrieves most relevant chunks
    for a given user question

    Process:
    Question → Embed → Search DB → Top K chunks
    """

    def __init__(self):
        self.embedder = Embedder()
        self.db = VectorDB()

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K
    ) -> list:
        """
        Find most relevant chunks for query

        Steps:
        1. Embed the user question
        2. Search ChromaDB for similar chunks
        3. Return top K results with scores
        """

        print(f"    🔍 Searching for: '{query[:50]}...'")

        # ✅ Step 1: Embed query
        query_embedding = self.embedder.embed_text(query)

        # ✅ Step 2: Search DB
        results = self.db.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        # ✅ Step 3: Filter low quality results
        filtered = [
            r for r in results
            if r["score"] > 0.3  # minimum relevance
        ]

        print(
            f"    ✅ Found {len(filtered)} relevant chunks "
            f"(threshold: 0.3)"
        )

        return filtered

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = TOP_K
    ) -> dict:
        """
        Retrieve chunks and build context string
        Ready to pass to LLM

        Returns:
        {
            chunks: [...],
            context: "combined text",
            sources: ["file1.pdf", ...]
        }
        """

        chunks = self.retrieve(query, top_k)

        if not chunks:
            return {
                "chunks": [],
                "context": "",
                "sources": [],
                "found": False
            }

        # ✅ Build context from chunks
        context_parts = []
        sources = set()

        for i, chunk in enumerate(chunks, 1):
            source = chunk["metadata"].get("source", "unknown")
            score = round(chunk["score"], 3)
            sources.add(source)

            context_parts.append(
                f"[Chunk {i} | Source: {source} | "
                f"Relevance: {score}]\n"
                f"{chunk['text']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        return {
            "chunks": chunks,
            "context": context,
            "sources": list(sources),
            "found": True
        }

    def get_db_stats(self) -> dict:
        """Get database statistics"""
        return self.db.get_stats()