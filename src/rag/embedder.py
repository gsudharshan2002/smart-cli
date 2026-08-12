from src.rag.config import EMBEDDING_MODEL


class Embedder:
    """
    Creates vector embeddings using
    local sentence-transformers model

    FREE - No API key needed!
    Model: all-MiniLM-L6-v2
    Dimensions: 384
    Size: ~80MB (downloaded once)
    """
    def __init__(self):
        self.model = None
        self.model_name = EMBEDDING_MODEL

    def load_model(self):
        """Load embedding model (once)"""
        if self.model is None:
            print(
                f"    🤖 Loading embedding model: "
                f"{self.model_name}"
            )
            print(
                "    ⏳ First time may take 1-2 mins "
                "(downloading ~80MB)..."
            )

            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)

            print("    ✅ Embedding model loaded!")

        return self.model

    def embed_text(self, text: str) -> list:
        """
        Convert text to vector embedding

        Input:  "Hello world"
        Output: [0.1, 0.3, -0.2, ...] (384 numbers)
        """
        model = self.load_model()
        embedding = model.encode(text, show_progress_bar=False)
        return embedding.tolist()

    def embed_texts(self, texts: list) -> list:
        """
        Convert multiple texts to embeddings
        Much faster than one by one!
        """
        model = self.load_model()

        print(
            f"    🔢 Embedding {len(texts)} chunks..."
        )

        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True
        )

        print(
            f"    ✅ Created {len(embeddings)} embeddings "
            f"of dimension {len(embeddings[0])}"
        )

        return [e.tolist() for e in embeddings]

    def embed_chunks(self, chunks: list) -> list:
        """
        Add embeddings to chunks
        Returns chunks with embeddings added
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embed_texts(texts)

        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]

        return chunks