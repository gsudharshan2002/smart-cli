from src.rag.embedder import Embedder
from src.rag.vectordb import VectorDB
from src.rag.config import TOP_K


def build_context(chunks: list) -> str:
    """
    Turn retrieved chunks into the context block for the LLM.

    Each chunk line:
    [chunk_id: X | Source: file.pdf | Relevance: 0.87]
    <chunk text>
    """
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source", "unknown")
        chunk_id = chunk["metadata"].get("chunk_id")
        score = round(chunk["score"], 3)
        context_parts.append(
            f"[chunk_id: {chunk_id} | Source: {source} | "
            f"Relevance: {score}]\n"
            f"{chunk['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


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
        top_k: int = TOP_K,
        where: dict = None
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
            top_k=top_k,
            where=where
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
        top_k: int = TOP_K,
        where: dict = None
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

        chunks = self.retrieve(query, top_k, where=where)

        if not chunks:
            return {
                "chunks": [],
                "context": "",
                "sources": [],
                "found": False
            }

        # ✅ Build context from chunks
        context = build_context(chunks)
        sources = list(set(
            chunk["metadata"].get("source", "unknown")
            for chunk in chunks
        ))

        return {
            "chunks": chunks,
            "context": context,
            "sources": sources,
            "found": True
        }

    def get_db_stats(self) -> dict:
        """Get database statistics"""
        return self.db.get_stats()