import chromadb
from chromadb.config import Settings
from src.rag.config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    TOP_K
)



class VectorDB:
    """
    Manages ChromaDB vector database

    Stores:   Document chunks + embeddings
    Searches: Find similar chunks by vector
    Persists: Saves to disk at chroma_db/
    """

    def __init__(self):
        self.client = None
        self.collection = None

    def connect(self):
        """Connect to ChromaDB"""
        if self.client is None:
            print(
                f"    🗄️  Connecting to ChromaDB at: "
                f"{CHROMA_DB_PATH}"
            )

            self.client = chromadb.PersistentClient(
                path=CHROMA_DB_PATH
            )

            print("    ✅ ChromaDB connected!")

        return self.client

    def get_collection(self, name: str = COLLECTION_NAME):
        """Get or create collection"""
        client = self.connect()

        self.collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )

        print(
            f"    📁 Collection: '{name}' "
            f"({self.collection.count()} docs)"
        )

        return self.collection

    def add_chunks(self, chunks: list):
        """
        Add chunks with embeddings to ChromaDB

        Each chunk needs:
        - id: unique string
        - text: the text content
        - embedding: vector list
        - metadata: dict of info
        """
        collection = self.get_collection()

        # ✅ Prepare data for ChromaDB
        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk["id"])
            documents.append(chunk["text"])
            embeddings.append(chunk["embedding"])

            # ✅ Clean metadata
            # ChromaDB only accepts str, int, float, bool
            clean_meta = {}
            for k, v in chunk["metadata"].items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)

            metadatas.append(clean_meta)

        # ✅ Add to ChromaDB in batches
        batch_size = 100
        total = len(ids)

        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)

            collection.upsert(
                ids=ids[i:batch_end],
                documents=documents[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )

            print(
                f"    💾 Stored chunks "
                f"{i+1}-{batch_end} of {total}"
            )

        print(
            f"    ✅ Total in DB: "
            f"{collection.count()} chunks"
        )

    def search(
        self,
        query_embedding: list,
        top_k: int = TOP_K,
        where: dict = None
    ) -> list:
        """
        Find most similar chunks to query

        Returns top_k most relevant chunks
        """
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.get_collection().count()),
            "include" :["documents", "metadatas", "distances"]
        }

        if where :
            query_kwargs["where"] = where
        
        results = self.get_collection().query(**query_kwargs)
        

        

        # ✅ Format results
        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]
            })

        return chunks

    def get_stats(self) -> dict:
        """Get database statistics"""
        collection = self.get_collection()
        count = collection.count()

        return {
            "total_chunks": count,
            "collection": COLLECTION_NAME,
            "db_path": CHROMA_DB_PATH
        }

    def delete_collection(self):
        """Delete all data from collection"""
        client = self.connect()
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"    🗑️  Collection deleted!")
        except Exception:
            print("    ℹ️  No collection to delete")

    def document_exists(self, source: str) -> bool:
        """Check if document already indexed"""
        collection = self.get_collection()

        results = collection.get(
            where={"source": source}
        )

        return len(results["ids"]) > 0