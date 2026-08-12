from src.rag.config import CHUNK_SIZE, CHUNK_OVERLAP


class TextChunker:
    """
    Splits large documents into
    smaller overlapping chunks
    for better retrieval
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(
        self,
        text: str,
        metadata: dict
    ) -> list:
        """
        Split text into overlapping chunks

        Example:
        Text: "AAABBBCCC"
        chunk_size=3, overlap=1
        Chunks: ["AAA", "ABC", "BCC", "CCC"]
        """

        if not text:
            return []

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            # ✅ Get chunk end position
            end = start + self.chunk_size

            # ✅ Try to end at sentence boundary
            if end < len(text):
                # Find last period or newline
                last_period = text.rfind(".", start, end)
                last_newline = text.rfind("\n", start, end)
                boundary = max(last_period, last_newline)

                if boundary > start + (self.chunk_size // 2):
                    end = boundary + 1

            # ✅ Extract chunk
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "id": f"{metadata.get('source', 'doc')}_{chunk_id}",
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_id": chunk_id,
                        "chunk_start": start,
                        "chunk_end": end,
                        "chunk_size": len(chunk_text)
                    }
                })
                chunk_id += 1

            # ✅ Move forward with overlap
            start = end - self.overlap

            if start >= len(text):
                break

        return chunks

    def chunk_document(self, document: dict) -> list:
        """Chunk a loaded document"""
        text = document.get("text", "")
        metadata = document.get("metadata", {})

        chunks = self.chunk_text(text, metadata)

        print(
            f"    ✂️  '{metadata.get('source', 'doc')}' "
            f"→ {len(chunks)} chunks "
            f"(size={self.chunk_size}, "
            f"overlap={self.overlap})"
        )

        return chunks

    def chunk_all_documents(
        self,
        documents: list
    ) -> list:
        """Chunk all loaded documents"""
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        print(
            f"\n    📊 Total chunks: {len(all_chunks)} "
            f"from {len(documents)} documents"
        )

        return all_chunks