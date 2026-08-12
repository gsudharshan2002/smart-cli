# src/features/rag_chat.py - RAG CLI Feature

from src.rag.rag_engine import RAGEngine
from src.utils.printer import (
    print_feature_header,
    print_concept,
    print_response,
    type_response,
    print_step,
    print_thinking,
    print_info,
    print_divider,
    print_prompt,
    print_success,
    print_error,
    print_table
)
from src.utils.menu import get_user_input
from rich.console import Console

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


def chat_with_documents(engine: RAGEngine):
    """Interactive chat with documents"""

    # ✅ Check DB has data
    stats = engine.get_stats()

    if stats["total_chunks"] == 0:
        print_error(
            "No documents indexed yet!\n"
            "Please index documents first (Option 2)"
        )
        return

    print_success(
        f"Ready to chat! "
        f"{stats['total_chunks']} chunks indexed from "
        f"{len(stats['documents'])} documents"
    )

    print_info(
        "Ask any question about your documents!\n"
        "Type 'quit' to stop chatting\n"
        "Type 'sources' to see what's indexed"
    )

    print_divider()

    # ✅ Chat loop
    question_count = 0

    while True:
        question_count += 1

        question = get_user_input(
            f"\n💬 Question {question_count}: "
        )

        if not question:
            continue

        if question.lower() == "quit":
            print_success("Chat ended!")
            break

        if question.lower() == "sources":
            show_documents(engine)
            continue

        print_divider()
        print_prompt(f"Your Question:\n{question}")

        print_thinking()

        # ✅ Full RAG query
        result = engine.query(
            question=question,
            top_k=3
        )

        # ✅ Show answer (streamed like real AI chat)
        type_response(result["answer"], title="RAG Answer")

        # ✅ Show sources
        print_info(
            f"📚 Sources used: "
            f"{', '.join(result['sources'])}\n"
            f"🔍 Chunks retrieved: {result['chunks_used']}"
        )

        # ✅ Show retrieved chunks detail
        show_chunks = get_user_input(
            "Show retrieved chunks? (yes/no): "
        )

        if show_chunks.lower() in ["yes", "y"]:
            for i, chunk in enumerate(
                result.get("chunks", []), 1
            ):
                print_info(
                    f"Chunk {i}:\n"
                    f"  Source: {chunk['metadata'].get('source')}\n"
                    f"  Score:  {round(chunk['score'], 3)}\n"
                    f"  Text:   {chunk['text'][:300]}..."
                )

        print_divider()


def run():
    """RAG Chat Feature"""

    # ✅ Header
    print_feature_header("RAG — Document Q&A")

    # ✅ Explain concept
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
        "  Find Similar Chunks\n"
        "      ↓\n"
        "  Send to LLM with Context\n"
        "      ↓\n"
        "  Accurate Answer from YOUR docs!\n\n"
        "Stack:\n"
        "  📄 Loader    → pypdf (PDF reader)\n"
        "  ✂️  Chunker   → Smart text splitter\n"
        "  🔢 Embedder  → all-MiniLM-L6-v2 (FREE)\n"
        "  🗄️  Vector DB → ChromaDB (LOCAL)\n"
        "  🤖 LLM       → Groq llama-3.3-70b"
    )

    print_divider()

    # ✅ Initialize RAG Engine
    print_step("Init", "Starting RAG Engine...")
    engine = RAGEngine()
    print_success("RAG Engine ready!")

    print_divider()

    # ✅ Main RAG Menu loop
    while True:
        console.print(
            "\n[bold cyan]RAG Options:[/bold cyan]\n"
            "  [green]1[/green] → Show documents\n"
            "  [green]2[/green] → Index documents\n"
            "  [green]3[/green] → Chat with documents\n"
            "  [green]4[/green] → Re-index a document\n"
            "  [green]5[/green] → Clear database\n"
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
                "📝 Enter document name to re-index: "
            )

            import os
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
                "⚠️  Delete ALL indexed data? (yes/no): "
            )

            if confirm.lower() in ["yes", "y"]:
                engine.db.delete_collection()
                print_success("Database cleared!")
            else:
                print_info("Cancelled!")

        else:
            print_error("Invalid option!")