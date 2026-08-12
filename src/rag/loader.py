import os
from pathlib import Path
from src.rag.config import DATA_PATH


class DocumentLoader:
    def __init__(self):
        self.data_path = DATA_PATH
        self.supported = [".pdf", ".txt", ".docx"]

    def list_documents(self) -> list:
        """List all documents in data/ folder"""
        docs = []

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            return docs

        for file in Path(self.data_path).iterdir():
            if file.suffix.lower() in self.supported:
                docs.append({
                    "name": file.name,
                    "path": str(file),
                    "type": file.suffix.lower(),
                    "size": self._get_size(file)
                })

        return docs

    def _get_size(self, path: Path) -> str:
        """Get human readable file size"""
        size = path.stat().st_size
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size // 1024} KB"
        else:
            return f"{size // (1024 * 1024)} MB"

    def load_pdf(self, path: str) -> str:
        """Load and extract text from PDF"""
        try:
            from pypdf import PdfReader

            reader = PdfReader(path)
            text = ""
            total_pages = len(reader.pages)

            print(f"    📄 Reading {total_pages} pages...")

            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n[Page {i+1}]\n{page_text}"

            return text.strip()

        except Exception as e:
            print(f"    ❌ PDF Error: {e}")
            return ""

    def load_txt(self, path: str) -> str:
        """Load plain text file"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"    ❌ TXT Error: {e}")
            return ""
    def load_docx(self, path: str) -> str:
        """Load Word document"""
        try:
            import docx
            doc = docx.Document(path)
            text = ""
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"
            return text.strip()
        except Exception as e:
            print(f"    ❌ DOCX Error: {e}")
            return ""

    def load_document(self, path: str) -> dict:
        """
        Load any supported document
        Returns dict with text and metadata
        """
        file = Path(path)
        ext = file.suffix.lower()

        print(f"    📂 Loading: {file.name}")

        if ext == ".pdf":
            text = self.load_pdf(path)
        elif ext == ".txt":
            text = self.load_txt(path)
        elif ext == ".docx":
            text = self.load_docx(path)
        else:
            return {
                "text": "",
                "metadata": {},
                "error": f"Unsupported: {ext}"
            }

        return {
            "text": text,
            "metadata": {
                "source": file.name,
                "path": str(file),
                "type": ext,
                "size": self._get_size(file),
                "chars": len(text),
                "words": len(text.split())
            }
        }

    def load_all(self) -> list:
        """Load all documents from data/ folder"""
        docs = self.list_documents()
        loaded = []

        for doc in docs:
            result = self.load_document(doc["path"])
            if result["text"]:
                loaded.append(result)

        return loaded