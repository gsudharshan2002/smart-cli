# src/features/rag_chat.py - RAG CLI Feature (Split-View Chat)

from src.rag.rag_engine import RAGEngine
from src.features.retrieval_lab import run as run_lab
from src.utils.printer import (
    print_feature_header,
    print_concept,
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
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.align import Align
from rich.console import Group
import time
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


# ─────────────────────────────────────────────────────────────────────
# Split-View Chat Components
# ─────────────────────────────────────────────────────────────────────

class ChatSession:
    """Manages chat history and context for the split-view UI."""

    def __init__(self, engine: RAGEngine):
        self.engine = engine
        self.stats = engine.get_stats()
        self.history = []  # list of (question, answer, result_dict)
        self.question_count = 0
        self.current_result = None
        self.show_context = True

    def build_chat_panel(self) -> Panel:
        """Build the left panel showing conversation history."""
        if not self.history:
            content = Text("Welcome to RAG Chat!\n\n")
            content.append("Type your question below.\n", style="dim")
            content.append("Commands: quit, context, sources, clear", style="dim cyan")
            return Panel(
                content,
                title="💬 Chat",
                border_style="cyan",
                padding=(1, 2)
            )

        lines = []
        for i, (q, a, _) in enumerate(self.history, 1):
            lines.append(Text(f"Q{i}: ", style="bold green"))
            lines.append(Text(f"{q}\n", style="white"))
            lines.append(Text(f"A{i}: ", style="bold magenta"))
            lines.append(Text(f"{a}\n\n", style="white"))

        return Panel(
            Group(*lines),
            title=f"💬 Chat ({len(self.history)} messages)",
            border_style="cyan",
            padding=(1, 2)
        )

    def build_context_panel(self) -> Panel:
        """Build the right panel showing retrieval context."""
        if not self.current_result:
            content = Text("Ask a question to see retrieval context here.", style="dim")
            return Panel(
                content,
                title="🔍 Retrieval Context",
                border_style="yellow",
                padding=(1, 2)
            )

        chunks = self.current_result.get("chunks", [])
        if not chunks:
            content = Text("No chunks retrieved.", style="dim red")
            return Panel(
                content,
                title="🔍 Retrieval Context",
                border_style="yellow",
                padding=(1, 2)
            )

        # Build context table
        table = Table(show_header=True, header_style="bold yellow", box=None)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Source", style="white", width=25)
        table.add_column("Vector", justify="right", width=8)
        table.add_column("BM25#", justify="right", width=6)
        table.add_column("RRF#", justify="right", width=6)
        table.add_column("Rerank", justify="right", width=8)

        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            source = meta.get("source", "?")
            if len(source) > 24:
                source = source[:21] + "..."

            vector_score = chunk.get("vector_score")
            vector_str = f"{vector_score:.3f}" if vector_score is not None else "-"

            bm25_rank = chunk.get("bm25_rank")
            bm25_str = str(bm25_rank) if bm25_rank is not None else "-"

            rrf_rank = chunk.get("rrf_rank")
            rrf_str = str(rrf_rank) if rrf_rank is not None else "-"

            rerank_score = chunk.get("rerank_score")
            rerank_str = f"{rerank_score:.3f}" if rerank_score is not None else "-"

            table.add_row(str(i), source, vector_str, bm25_str, rrf_str, rerank_str)

        # Sources summary
        sources = self.current_result.get("sources", [])
        chunks_used = self.current_result.get("chunks_used", 0)

        summary = Text()
        summary.append(f"📚 Sources: {', '.join(sources)}\n", style="white")
        summary.append(f"🔍 Chunks retrieved: {chunks_used}\n", style="dim")
        if self.current_result.get("retrieval", {}).get("search_query"):
            sq = self.current_result["retrieval"]["search_query"]
            if sq != self.history[-1][0] if self.history else True:
                summary.append(f"🔄 Rewritten: {sq[:60]}...", style="dim cyan")

        content = Group(
            Panel(table, title="Retrieved Chunks", border_style="dim", padding=(0, 1)),
            summary
        )

        return Panel(
            content,
            title="🔍 Retrieval Context",
            border_style="yellow",
            padding=(1, 2)
        )

    def build_input_panel(self) -> Panel:
        """Build the bottom input hint panel."""
        return Panel(
            Text.from_markup(
                "[bold green]Your question:[/bold green]  (type 'quit' to exit, 'context' to toggle side panel)"
            ),
            border_style="green",
            padding=(0, 1)
        )

    def render(self):
        """Render the full split-view UI by clearing and redrawing."""
        console.clear()

        # Header
        console.print(
            Panel(
                Text("RAG Split-View Chat", style="bold cyan"),
                border_style="cyan"
            )
        )

        # Main split row
        chat_panel = self.build_chat_panel()
        if self.show_context:
            context_panel = self.build_context_panel()
            # Use Columns for side-by-side layout
            console.print(Columns([chat_panel, context_panel], equal=False, expand=True))
        else:
            console.print(chat_panel)

        # Input hint
        console.print(self.build_input_panel())
        console.print()  # spacing


def chat_with_documents(engine: RAGEngine):
    """Interactive split-view chat with documents."""

    # Check DB has data
    stats = engine.get_stats()

    if stats["total_chunks"] == 0:
        print_error(
            "No documents indexed yet!\n"
            "Please index documents first (Option 2)"
        )
        return

    session = ChatSession(engine)

    print_success(
        f"Ready to chat! {stats['total_chunks']} chunks from {len(stats['documents'])} docs\n"
        f"Retrieval: hybrid (vector + BM25, RRF) + cross-encoder rerank"
    )
    print_info("Commands: quit | context (toggle) | sources | clear")
    print_divider()

    while True:
        session.render()

        question = get_user_input("💬 Question: ").strip()

        if not question:
            continue

        if question.lower() == "quit":
            print_success("Chat ended!")
            break

        if question.lower() == "context":
            session.show_context = not session.show_context
            continue

        if question.lower() == "sources":
            show_documents(engine)
            print_divider()
            continue

        if question.lower() == "clear":
            session.history.clear()
            session.current_result = None
            continue

        session.question_count += 1

        # Show thinking indicator
        print_thinking()

        # Full RAG query
        result = engine.query(question=question, top_k=3)

        # Store in history
        session.current_result = result
        session.history.append((question, result["answer"], result))

        # Loop continues - screen will be cleared and redrawn with new history


def run():
    """RAG Chat Feature with Split-View"""

    # Header
    print_feature_header("RAG — Document Q&A (Split-View)")

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
        "UI: Split-view — Chat on left, Retrieval context on right"
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
            "  [green]3[/green] → Chat with documents (split-view)\n"
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