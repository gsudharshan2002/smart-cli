# test_rag_setup.py

from src.rag.loader import DocumentLoader
from src.rag.chunker import TextChunker
from src.rag.embedder import Embedder
from src.rag.vectordb import VectorDB

print("\n🔍 Step 1: Loading documents...")
loader = DocumentLoader()
docs = loader.list_documents()
print(f"Found {len(docs)} documents:")
for d in docs:
    print(f"  📄 {d['name']} ({d['size']})")

print("\n✂️  Step 2: Loading & Chunking...")
loaded = loader.load_all()
chunker = TextChunker()
chunks = chunker.chunk_all_documents(loaded)

print(f"\n🔢 Step 3: Embedding chunks...")
embedder = Embedder()
chunks_with_embeddings = embedder.embed_chunks(chunks)
print(
    f"First chunk embedding size: "
    f"{len(chunks_with_embeddings[0]['embedding'])}"
)

print(f"\n💾 Step 4: Storing in ChromaDB...")
db = VectorDB()
db.add_chunks(chunks_with_embeddings)

print(f"\n🔎 Step 5: Test search...")
query = "What is this document about?"
query_embedding = embedder.embed_text(query)
results = db.search(query_embedding, top_k=3)

print(f"\nTop {len(results)} results:")
for i, r in enumerate(results, 1):
    print(f"\n  Result {i}:")
    print(f"  Source: {r['metadata'].get('source')}")
    print(f"  Score:  {round(r['score'], 3)}")
    print(f"  Text:   {r['text'][:150]}...")

print("\n✅ RAG Setup working!")