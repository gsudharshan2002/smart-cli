import re

from src.rag.config import CHUNK_SIZE, CHUNK_OVERLAP

PAGE_MARKER_RE = re.compile(r"\[Page (\d+)\]")


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

        page_markers = [
            (m.start(), int(m.group(1)))
            for m in PAGE_MARKER_RE.finditer(text)
        ]

        def page_for(offset: int) -> int:
            """Page number of the last marker at or before `offset`."""
            page_number = 1
            for pos, num in page_markers:
                if pos <= offset:
                    page_number = num
                else:
                    break
            return page_number

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
                        "chunk_size": len(chunk_text),
                        "page": page_for(start),
                        "strategy": "fixed"
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


class StructureChunker:
    """
    Splits text along document structure (numbered headings) instead
    of fixed character windows — one chunk per section, e.g. "5.2
    Build an Internal Personal Assistant Chatbot" becomes one chunk.
    """

    HEADING_RE = re.compile(r"^(\d+(?:\.\d+)?)\.?\s+([A-Z].+)$", re.MULTILINE)

    def chunk_text(self, text: str, metadata: dict) -> list:
        if not text:
            return []

        source = metadata.get("source", "doc")
        headings = list(self.HEADING_RE.finditer(text))

        page_markers = [
            (m.start(), int(m.group(1)))
            for m in PAGE_MARKER_RE.finditer(text)
        ]

        def page_for(offset: int) -> int:
            page_number = 1
            for pos, num in page_markers:
                if pos <= offset:
                    page_number = num
                else:
                    break
            return page_number

        item_start = re.compile(r"^(Phase \d|Ongoing:)")

        chunks = []
        cid = 0  # counts CHUNKS produced, not headings visited

        for idx, m in enumerate(headings):
            sec_end = headings[idx + 1].start() if idx + 1 < len(headings) else len(text)
            heading_title = f"{m.group(1)} {m.group(2).strip()}"
            body = text[m.end():sec_end].strip()
            page = page_for(m.start())

            if not body:
                continue

            # Merge PDF-wrapped continuation lines back into one line per bullet
            raw_lines = [l.strip() for l in body.split("\n") if l.strip()]
            logical_lines = []
            for l in raw_lines:
                if item_start.match(l) or not logical_lines:
                    logical_lines.append(l)
                else:
                    logical_lines[-1] += " " + l

            if logical_lines and all(item_start.match(l) for l in logical_lines):
                # Bulleted section (e.g. the roadmap) -> one chunk per bullet
                for bullet in logical_lines:
                    chunks.append({
                        "id": f"{source}_structure_{cid}",
                        "text": bullet,
                        "metadata": {
                            "source": source,
                            "section": heading_title,
                            "page": page,
                            "strategy": "structure",
                            "chunk_id": cid,
                            "chunk_size": len(bullet),
                        }
                    })
                    cid += 1
            else:
                # Normal section -> one chunk for the whole body
                chunk_text_ = f"{heading_title}\n{body}"
                chunks.append({
                    "id": f"{source}_structure_{cid}",
                    "text": chunk_text_,
                    "metadata": {
                        "source": source,
                        "section": heading_title,
                        "page": page,
                        "strategy": "structure",
                        "chunk_id": cid,
                        "chunk_size": len(chunk_text_),
                    }
                })
                cid += 1

        return chunks

    def chunk_document(self, document: dict) -> list:
        text = document.get("text", "")
        metadata = document.get("metadata", {})
        chunks = self.chunk_text(text, metadata)
        print(f"    ✂️  [structure] '{metadata.get('source', 'doc')}' → {len(chunks)} chunks")
        return chunks
