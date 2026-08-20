# src/features/rag_chat.py - RAG CLI Feature (Plain CLI Chat)

from src.rag.rag_engine import RAGEngine
from src.features.retrieval_lab import run as run_lab
from src.utils.printer import (
    print_feature_header,
    print_concept,
    print_step,
    print_thinking,
    print_info,
    print_divider,
    print_success,
    print_error,
    print_table,
    type_response
)
from src.utils.menu import get_user_input
from rich.console import Console
import os

console = Console()


def show_documents(engine: RAGEngine):
    """Show all documents and their status"""

    stats = engine.get_stats()
    docs = stats["documents"]

    if not docs:
        print_error(
            "No documents found in data/ folder!\n"
            "Add PDF/TXT/DOCX files to data/ folder"
        )
        return False

    rows = []
    for doc in docs:
        rows.append([
            doc["name"],
            doc["size"],
            "✅ Indexed" if doc["indexed"] else "⏳ Not indexed"
        ])

    print_table(
        title="📚 Documents in data/ folder",
        columns=["Document", "Size", "Status"],
        rows=rows
    )

    print_info(
        f"Total chunks in DB: {stats['total_chunks']}"
    )

    return True


def index_documents(engine: RAGEngine):
    """Index all documents"""

    print_step(
        "Indexing",
        "Loading all documents from data/ folder..."
    )

    results = engine.index_all_documents()

    if not results:
        print_error("No documents to index!")
        return False

    success = sum(
        1 for r in results if r["status"] == "success"
    )
    skipped = sum(
        1 for r in results if r["status"] == "skipped"
    )

    print_success(
        f"Indexing complete!\n"
        f"  ✅ Indexed: {success} documents\n"
        f"  ⏭️  Skipped: {skipped} (already indexed)"
    )

    return True


def show_chunk_details(result: dict):
    """Print per-chunk retrieval details + latency for the last answer."""
    chunks = result.get("chunks", [])
    if not chunks:
        print_info("No chunk details available for the last answer.")
        return

    rows = []
    for c in chunks:
        meta = c.get("metadata", {})

        def fmt(v):
            if isinstance(v, float):
                return f"{v:.3f}"
            return "-" if v is None else str(v)

        rows.append([
            str(len(rows) + 1),
            str(meta.get("chunk_id", "?")),
            str(meta.get("source", "?")),
            str(meta.get("page", "?")),
            fmt(c.get("vector_score")),
            fmt(c.get("bm25_rank")),
            fmt(c.get("rrf_rank")),
            fmt(c.get("rerank_score")),
            c["text"][:55].replace("\n", " "),
        ])

    print_table(
        title="Retrieved chunks (last answer)",
        columns=["#", "ChunkID", "Source", "Page", "Vector", "BM25#", "RRF#", "Rerank", "Snippet"],
        rows=rows
    )

    trace = result.get("trace", {})
    if trace:
        stages = trace.get("stages", {})
        ret_s = stages.get("retrieval", {}).get("duration_s") or 0
        llm_s = stages.get("llm", {}).get("duration_s") or 0
        print_info(
            f"Request {trace.get('request_id')} — "
            f"retrieval {ret_s * 1000:.1f} ms, "
            f"LLM {llm_s * 1000:.1f} ms, "
            f"total {trace.get('total_duration_s', 0) * 1000:.1f} ms, "
            f"model calls {trace.get('total_model_calls', 0)}"
        )


def chat_with_documents(engine: RAGEngine):
    """Interactive chat with documents (plain CLI)."""

    stats = engine.get_stats()

    if stats["total_chunks"] == 0:
        print_error(
            "No documents indexed yet!\n"
            "Please index documents first (Option 2)"
        )
        return

    history = []
    last_result = None

    print_success(
        f"Ready to chat! {stats['total_chunks']} chunks from {len(stats['documents'])} docs\n"
        f"Retrieval: hybrid (vector + BM25, RRF) + cross-encoder rerank"
    )
    print_info("Commands: quit | details | sources | clear")
    print_divider()

    while True:
        question = get_user_input("💬 Question: ").strip()

        if not question:
            continue

        if question.lower() == "quit":
            print_success("Chat ended!")
            break

        if question.lower() == "details":
            if last_result:
                show_chunk_details(last_result)
            else:
                print_info("Ask a question first, then type 'details'.")
            print_divider()
            continue

        if question.lower() == "sources":
            show_documents(engine)
            print_divider()
            continue

        if question.lower() == "clear":
            history.clear()
            last_result = None
            print_info("History cleared!")
            continue

        # Show thinking indicator
        print_thinking()

        # Full RAG query
        result = engine.query(question=question, top_k=3)

        # Stream the answer like a typewriter
        type_response(result["answer"], title="Answer")

        last_result = result
        history.append((question, result["answer"]))

        if result["sources"]:
            print_info(
                f"Sources: {', '.join(result['sources'])}"
            )
        print_info("Type 'details' to see chunk_id / page / scores / latency")


def run():
    """RAG Chat Feature with Plain CLI"""

    # Header
    print_feature_header("RAG — Document Q&A")

    # Explain concept
    print_concept(
        "What is RAG?",
        "RAG = Retrieval Augmented Generation\n\n"
        "How it works:\n\n"
        "INDEX (done once):\n"
        "  Your PDF\n"
        "      ↓\n"
        "  Extract Text\n"
        "      ↓\n"
        "  Split into Chunks\n"
        "      ↓\n"
        "  Create Embeddings (vectors)\n"
        "      ↓\n"
        "  Store in ChromaDB\n\n"
        "QUERY (every question):\n"
        "  Your Question\n"
        "      ↓\n"
        "  Embed Question\n"
        "      ↓\n"
        "  Find Similar Chunks (hybrid + rerank)\n"
        "      ↓\n"
        "  Send to LLM with Context\n"
        "      ↓\n"
        "  Accurate Answer from YOUR docs!\n\n"
        "Stack:\n"
        "  Loader    -> pypdf (PDF reader)\n"
        "  Chunker   -> Smart text splitter\n"
        "  Embedder  -> all-MiniLM-L6-v2 (free)\n"
        "  VectorDB  -> ChromaDB (local)\n"
        "  Keyword   -> BM25 + RRF hybrid fusion\n"
        "  Reranker  -> cross-encoder/ms-marco-MiniLM-L-6-v2\n"
        "  LLM       -> Groq gpt-oss-20b\n\n"
        "UI: Plain CLI — type your question, get the answer streamed"
    )

    print_divider()

    # Initialize RAG Engine
    print_step("Init", "Starting RAG Engine...")
    engine = RAGEngine()
    print_success("RAG Engine ready!")

    print_divider()

    # Main RAG Menu loop
    while True:
        console.print(
            "\n[bold cyan]RAG Options:[/bold cyan]\n"
            "  [green]1[/green] → Show documents\n"
            "  [green]2[/green] → Index documents\n"
            "  [green]3[/green] → Chat with documents\n"
            "  [green]4[/green] → Re-index a document\n"
            "  [green]5[/green] → Clear database\n"
            "  [green]6[/green] → Retrieval Lab (dev tools)\n"
            "  [green]0[/green] → Back to main menu\n"
        )

        choice = get_user_input("Choose option: ")

        if choice == "0":
            break

        elif choice == "1":
            print_divider()
            show_documents(engine)

        elif choice == "2":
            print_divider()
            print_step(
                "Indexing",
                "This may take a few minutes first time..."
            )
            index_documents(engine)

        elif choice == "3":
            print_divider()
            chat_with_documents(engine)

        elif choice == "4":
            print_divider()
            show_documents(engine)

            doc_name = get_user_input(
                "Enter document name to re-index: "
            )

            from src.rag.config import DATA_PATH
            doc_path = os.path.join(DATA_PATH, doc_name)

            if os.path.exists(doc_path):
                result = engine.reindex_document(doc_path)
                if result["status"] == "success":
                    print_success(
                        f"Re-indexed: {doc_name}\n"
                        f"Chunks: {result['chunks']}"
                    )
                else:
                    print_error(
                        f"Error: {result.get('reason')}"
                    )
            else:
                print_error(f"File not found: {doc_name}")

        elif choice == "5":
            print_divider()

            confirm = get_user_input(
                "Delete ALL indexed data? (yes/no): "
            )

            if confirm.lower() in ["yes", "y"]:
                engine.db.delete_collection()
                print_success("Database cleared!")
            else:
                print_info("Cancelled!")

        elif choice == "6":
            print_divider()
            run_lab(engine)

        else:
            print_error("Invalid option!")