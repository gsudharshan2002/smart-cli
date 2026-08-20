# src/features/retrieval_lab.py - Retrieval Lab (Evaluation & Inspection)

"""
Dev tools for the MERGED RAG pipeline.

The main RAG chat now runs hybrid retrieval (vector + BM25 + RRF +
cross-encoder rerank) via RAGEngine. This lab is where you:

  1. Learn the concepts
  2. Inspect what retrieval actually fetched for a question
  3. Measure before/after numbers (the PR evidence)
  4. Compare every retrieval variant
  5. Label the failures (wrong doc vs wrong answer)
"""

from src.rag.evaluator import Evaluator
from src.rag.hybrid import SmartRetriever
from src.rag.rag_engine import RAGEngine
from src.utils.printer import (
    print_feature_header,
    print_concept,
    print_success,
    print_error,
    print_info,
    print_divider,
    print_step,
    print_thinking,
    print_table
)
from src.utils.menu import get_user_input
from rich.console import Console

console = Console()


# ------------------------------------------------------------------
# 1. Concepts
# ------------------------------------------------------------------

def show_concepts():
    """Teach the retrieval inspection concepts."""
    print_concept(
        "Why not just use a smarter AI model?",
        "Retrieval failures are NOT model failures.\n\n"
        "If the app fetched the wrong document:\n"
        "  - a smarter model still sees the wrong document\n"
        "  - you'd pay more and fix nothing\n"
        "  - the fix lives in RETRIEVAL, not generation\n\n"
        "First you must find out WHICH kind of failure you have."
    )

    print_concept(
        "The two kinds of failure",
        "1) WRONG DOCUMENT FETCHED (retrieval failure)\n"
        "   The correct document never reached the LLM.\n"
        "   Evidence: answer cites the wrong source, or the\n"
        "   retrieved chunks are all from another policy.\n"
        "   Fix: better retrieval (BM25, hybrid, rerank).\n\n"
        "2) RIGHT DOCUMENT, WRONG ANSWER (generation failure)\n"
        "   The correct chunk WAS retrieved, but the LLM\n"
        "   answered badly (missed the number, hallucinated).\n"
        "   Evidence: correct source in top-k, wrong answer.\n"
        "   Fix: better prompting, better chunking, or a\n"
        "   stronger model.\n\n"
        "Label each failure, then fix with EXACTLY ONE change, "
        "then MEASURE before and after."
    )

    print_concept(
        "Keyword search (BM25)",
        "Embeddings catch MEANING; they miss EXACT TERMS.\n\n"
        "Ask: 'max nightly hotel rate for international travel?'\n"
        "  Embeddings → any 'hotel/travel' chunk\n"
        "  BM25      → chunks containing '325', 'international'\n\n"
        "BM25 = smarter word counting:\n"
        "  - term frequency: 'mileage' x3 in a chunk matters\n"
        "  - document frequency: rare words ('ERR-4032') matter\n"
        "    more than common ones ('the')\n"
        "  - length normalization: big chunks don't win by size\n\n"
        "This is what catches codes, IDs, and numbers: $325,\n"
        "10 days, HR-POL-014, skill.md."
    )

    print_concept(
        "Hybrid search (RRF fusion)",
        "Combining two ranked lists that use DIFFERENT scales:\n\n"
        "  vector scores: 0.30 - 0.99 (cosine similarity)\n"
        "  BM25 scores:   0 - 20 (raw term statistics)\n\n"
        "You can't average those numbers - they're apples and\n"
        "oranges. Reciprocal Rank Fusion avoids the problem:\n\n"
        "  score(chunk) = 1/(60+rank in list 1)\n"
        "               + 1/(60+rank in list 2)\n\n"
        "Only RANKS matter. #1 in both lists beats #1 in one.\n"
        "A chunk the keyword search found at #2 that the vector\n"
        "search missed entirely still scores high."
    )

    print_concept(
        "Reranking (cross-encoder)",
        "Bi-encoder (embeddings) vs cross-encoder:\n\n"
        "  Bi-encoder:    embed query and chunk SEPARATELY,\n"
        "                 compare with a dot product.\n"
        "                 Fast, pre-computable - but the model\n"
        "                 never sees them together.\n\n"
        "  Cross-encoder: feed (query + chunk) TOGETHER into\n"
        "                 the transformer. Slow, but sees the\n"
        "                 exact interaction → much better ranking.\n\n"
        "Use the cheap bi-encoder to pre-filter ~10 candidates,\n"
        "then the cross-encoder to re-rank them. Same idea as\n"
        "Cohere Rerank / BGE-Reranker, running locally and free."
    )

    print_concept(
        "MMR (Maximal Marginal Relevance)",
        "Diversity filter for a classic RAG failure:\n"
        "all 3 chunks from the same paragraph!\n\n"
        "  score = λ·relevance(query, doc)\n"
        "        - (1-λ)·max similarity(doc, already picked)\n\n"
        "  λ=1.0 → pure relevance, zero diversity\n"
        "  λ=0.0 → pure diversity\n\n"
        "Penalizes the 2nd chunk that repeats what the 1st\n"
        "already said. Use when answers are missing details\n"
        "that exist in the docs but in different sections."
    )

    print_concept(
        "Query rewriting & HyDE",
        "Fix the QUERY before search:\n\n"
        "  Rewriting:\n"
        "  'whats the max they give back if my hotel is pricey\n"
        "   while im abroad for work'\n"
        "    → 'maximum hotel reimbursement international travel'\n\n"
        "  HyDE (Hypothetical Document Embeddings):\n"
        "  Ask the LLM to write the ANSWER as it would appear in\n"
        "  the policy, then embed THAT instead of the question.\n"
        "  Question text and policy text use different vocabularies;\n"
        "  a fake answer and the real paragraph share one.\n\n"
        "Both are ONE extra LLM call per question."
    )

    print_concept(
        "Measuring: hit-rate@k, recall@k, MRR",
        "k = how many chunks you retrieve (e.g. top-3).\n\n"
        "  hit_rate@k : did the RIGHT DOCUMENT appear in top-k?\n"
        "               fraction of questions where yes.\n"
        "               THE number this week's task asks for.\n\n"
        "  recall@k   : did the exact right CHUNK appear in top-k?\n\n"
        "  mrr@k      : how EARLY was it? avg of 1/rank of first\n"
        "               right-document chunk. #1 → 1.0, #3 → 0.33.\n\n"
        "Measure BEFORE → make ONE change → measure AFTER.\n"
        "If the number didn't move, the change didn't help.\n"
        "And check WHICH failures your change did NOT fix."
    )


# ------------------------------------------------------------------
# 2. Inspection view
# ------------------------------------------------------------------

def _chunk_row(chunk: dict, rank: int) -> list:
    """One row for the inspection table."""
    meta = chunk["metadata"]
    snippet = chunk["text"][:90].replace("\n", " ")

    def fmt(name, width=7):
        v = chunk.get(name)
        return f"{v:.3f}" if isinstance(v, float) else ("-" if v is None else str(v))

    return [
        str(rank),
        str(meta.get("source", "?")),
        fmt("vector_score"),  # explicitly show vector score (cosine sim 0-1)
        fmt("bm25_rank") if "bm25_rank" in chunk else "-",
        fmt("rrf_rank") if "rrf_rank" in chunk else "-",
        fmt("rerank_score") if "rerank_score" in chunk else "-",
        snippet
    ]


def _show_retrieval_table(title: str, chunks: list):
    rows = [
        _chunk_row(c, i)
        for i, c in enumerate(chunks, 1)
    ]
    print_table(
        title=title,
        columns=["#", "Source", "Vector", "BM25#", "RRF#", "Rerank", "Snippet"],
        rows=rows
    )


def inspection_view(engine: RAGEngine, evaluator: Evaluator):
    """
    Ask one question and see it from both sides:
    - what the OLD (vector-only) retriever fetched
    - what the NEW (hybrid+rerank) retriever fetches
    - the final answer generated from the new chunks
    """
    print_step(
        "Inspection",
        "Question | What was fetched | Answer — side by side"
    )

    smart = evaluator.retriever

    while True:
        question = get_user_input(
            "\nQuestion to inspect (or 'quit'): "
        )

        if not question:
            continue
        if question.lower() == "quit":
            break

        print_divider()
        print_info(f"QUESTION: {question}")

        print_thinking()

        # BEFORE: the Week 4 retriever (vector only)
        vector_result = smart.retrieve(
            query=question,
            top_k=3,
            use_bm25=False,
            use_rerank=False
        )
        print_info("BEFORE -- vector search only (baseline):")
        _show_retrieval_table(
            "Old retriever: pure semantic (top-3)",
            vector_result["chunks"]
        )

        # AFTER: full merged pipeline
        full_result = smart.retrieve(
            query=question,
            top_k=3,
            use_bm25=True,
            use_rerank=True,
            use_rewrite=True
        )

        if full_result["warnings"]:
            for w in full_result["warnings"]:
                print_info(f"  info: {w}")

        print_info("AFTER -- hybrid + rerank + rewrite:")
        _show_retrieval_table(
            "New retriever: BM25 + RRF + cross-encoder (top-3)",
            full_result["chunks"]
        )

        # Generate the final answer from the improved chunks
        print_step("Answer", "Generating answer from the fetched chunks...")
        print_thinking()

        if full_result["chunks"]:
            answer = engine.answer_from_chunks(
                question, full_result["chunks"]
            )
            print_success("ANSWER (from what was fetched):")
            console.print(answer)
            print_divider()

            # Label this one
            label_now = get_user_input(
                "Classify this case? (1=wrong doc fetched, "
                "2=right doc/wrong answer, 3=not a failure, n=skip): "
            )
            if label_now in ["1", "2", "3"]:
                _save_label(question, label_now, evaluator, None)
        else:
            print_error("No chunks retrieved for this question!")


# ------------------------------------------------------------------
# 3. Before / After evaluation
# ------------------------------------------------------------------

def run_before_after(engine: RAGEngine, evaluator: Evaluator):
    """
    Measure baseline vs improved retrieval for PR evidence.
    1. measure baseline hit-rate@3 (vector only)
    2. make ONE change
    3. measure again, show the number, show what improved
    """
    top_k = 3

    print_step(
        "Measure",
        "Baseline first: hit-rate@3 with the vector-only retriever"
    )

    before = evaluator.evaluate_variant("vector", top_k=top_k)
    b = before["metrics"]
    print_success(
        f"BASELINE  hit-rate@{top_k}: {b['hit_rate']:.2%} "
        f"({b['hits']}/{b['questions']})  |  recall@{top_k}: "
        f"{b['recall']:.2%}  |  MRR@{top_k}: {b['mrr']:.3f}"
    )

    print_divider()
    print_concept(
        "Now pick EXACTLY ONE change",
        "The course rule: one change, so you know what helped.\n\n"
        "  1) hybrid        → add BM25 keyword search (RRF fusion)\n"
        "  2) hybrid+rerank → also add cross-encoder reranking\n"
        "  3) rewrite       → also rewrite messy questions first\n"
        "  4) hyde          → also embed a hypothetical answer\n"
        "\n"
        "Each is a SEPARATE experiment. Run one, read the number,\n"
        "run another. Never stack them all and wonder which helped."
    )

    choice = get_user_input(
        "Which ONE change to try? (1=hybrid, 2=+rerank, "
        "3=+rewrite, 4=hyde, 0=cancel): "
    )

    variant_map = {
        "1": "hybrid",
        "2": "hybrid+rerank",
        "3": "rewrite",
        "4": "hyde",
    }

    if choice not in variant_map:
        print_info("Cancelled!")
        return

    improved_variant = variant_map[choice]

    print_divider()
    print_step(
        "Measure again",
        f"hit-rate@{top_k} after change: '{improved_variant}'"
    )

    comparison = evaluator.compare(
        baseline="vector",
        improved=improved_variant,
        top_k=top_k
    )

    m = comparison["metrics"]
    print_divider()
    print_table(
        title=f"BEFORE / AFTER — hit-rate@{top_k} "
              f"(vector → {improved_variant})",
        columns=["Metric", "Before", "After", "Δ"],
        rows=[
            ["hit-rate@k",
             f"{m['hit_rate']['before']:.2%}",
             f"{m['hit_rate']['after']:.2%}",
             f"{m['hit_rate']['delta']:+.2%}"],
            ["recall@k",
             f"{m['recall']['before']:.2%}",
             f"{m['recall']['after']:.2%}",
             f"{m['recall']['delta']:+.2%}"],
            ["MRR@k",
             f"{m['mrr']['before']:.3f}",
             f"{m['mrr']['after']:.3f}",
             f"{m['mrr']['delta']:+.3f}"],
        ]
    )

    # Per-question diff
    rows = []
    for row in comparison["per_question"]:
        rows.append([
            f"Q{row['id']}",
            row["status"],
            row["question"][:50],
            row["expected_source"][:35],
        ])

    print_table(
        title="Per-question diff",
        columns=["Q#", "Status", "Question", "Expected doc"],
        rows=rows
    )

    # What did NOT get fixed
    still_failing = [
        r for r in comparison["per_question"]
        if not r["after_hit"]
    ]
    if still_failing:
        print_info(
            f"Still failing ({len(still_failing)} questions) - "
            "your change did NOT fix these:"
        )
        for r in still_failing:
            print_info(
                f"   Q{r['id']}: {r['question'][:70]}"
            )
        print_info(
            "Label them in option 5: wrong doc fetched, or "
            "right doc / wrong answer?"
        )
    else:
        print_success("All questions pass with this change!")


def compare_all_variants(engine: RAGEngine, evaluator: Evaluator):
    """Run every variant and compare - exploration only."""
    top_k = 3

    print_step(
        "Exploration",
        "Running all 5 variants over the eval set...\n"
        "(rewrite/hyde variants call the LLM once per question,\n"
        "this can take a minute or two)"
    )

    results = {}
    for variant in ["vector", "hybrid", "hybrid+rerank", "rewrite", "hyde"]:
        print_divider()
        r = evaluator.evaluate_variant(variant, top_k=top_k)
        results[variant] = r["metrics"]

    rows = []
    for variant, m in results.items():
        rows.append([
            variant,
            f"{m['hit_rate']:.2%}",
            f"{m['recall']:.2%}",
            f"{m['mrr']:.3f}",
            f"{m['hits']}/{m['questions']}"
        ])

    print_divider()
    print_table(
        title=f"All variants — hit-rate@{top_k} / recall@{top_k} / MRR@{top_k}",
        columns=["Variant", "hit-rate@k", "recall@k", "MRR@k", "hits"],
        rows=rows
    )
    print_info(
        "Read it like an experiment: each row adds ONE technique.\n"
        "The delta between rows tells you what each technique bought."
    )


# ------------------------------------------------------------------
# 4. Label the failures
# ------------------------------------------------------------------

def _save_label(question_text: str, kind: str, evaluator: Evaluator,
                question_id):
    kind_names = {
        "1": "wrong_document_fetched",
        "2": "right_document_wrong_answer",
        "3": "not_a_failure",
    }
    evaluator.save_label(
        question_id if question_id is not None else f"custom_{hash(question_text) % 100000}",
        kind_names[kind]
    )
    print_success(
        f"Label saved: {kind_names[kind]}"
    )


def label_failures(engine: RAGEngine, evaluator: Evaluator):
    """
    For every question the improved retriever still fails:
    show the evidence and let the user classify the failure kind.
    """
    top_k = 3

    print_step(
        "Labeling",
        "For each failing question: show the evidence (what was\n"
        "fetched + the generated answer), then classify the failure"
    )

    print_info(
        "Which retriever's failures do you want to label?\n"
        "  1 = baseline (vector only) — shows the ORIGINAL failures\n"
        "  2 = hybrid+rerank — what still fails after your change\n"
        "  3 = rewrite variant — the change that regressed"
    )
    choice = get_user_input("Choose (1/2/3): ")
    variant = {
        "1": "vector",
        "2": "hybrid+rerank",
        "3": "rewrite",
    }.get(choice, "hybrid+rerank")

    improved = evaluator.evaluate_variant(variant, top_k=top_k)

    failures = [r for r in improved["rows"] if not r["hit"]]

    if not failures:
        print_success(
            f"No retrieval failures with '{variant}'!"
        )
        print_info(
            "Try labeling the baseline (option 1) to see the "
            "original failures your change fixed."
        )
        return

    print_info(
        f"{len(failures)} question(s) still miss the right document.\n"
        "For each one we'll show: the question, what was fetched,\n"
        "and the generated answer - then you classify it."
    )

    smart = evaluator.retriever
    labels = evaluator.load_labels()

    for i, row in enumerate(failures, 1):
        print_divider()
        print_info(f"FAILURE {i}/{len(failures)} — Q{row['id']}")
        print_info(f"   Question:  {row['question']}")
        print_info(
            f"   Expected:  {row['expected_source']}"
        )
        print_info(
            f"   Retrieved: {', '.join(row['retrieved_sources'][:5])}"
        )

        # Show the chunks & generate the answer as evidence
        result = smart.retrieve(
            query=row["question"],
            top_k=top_k,
            use_bm25=True,
            use_rerank=True
        )
        _show_retrieval_table(
            f"Q{row['id']} -- what was fetched",
            result["chunks"]
        )

        print_thinking()
        answer = engine.answer_from_chunks(
            row["question"], result["chunks"]
        )
        console.print(f"[bold magenta]ANSWER:[/bold magenta] {answer}")

        print_info(
            "Classify: 1 = wrong document fetched\n"
            "          2 = right document, wrong answer\n"
            "          3 = not actually a failure (eval label wrong)"
        )

        qid = row["id"]
        already = labels.get(str(qid))
        if already:
            print_info(
                f"   (already labeled: {already['failure_kind']})"
            )

        choice = get_user_input("Your classification (1/2/3, n=skip): ")
        if choice in ["1", "2", "3"]:
            evaluator.save_label(qid, {
                "1": "wrong_document_fetched",
                "2": "right_document_wrong_answer",
                "3": "not_a_failure",
            }[choice])
            print_success("Saved!")

    print_divider()
    labels = evaluator.load_labels()
    kinds = {}
    for v in labels.values():
        kinds[v["failure_kind"]] = kinds.get(v["failure_kind"], 0) + 1
    print_table(
        title="Label summary (saved to data/eval_labels.json)",
        columns=["Failure kind", "Count"],
        rows=[list(x) for x in kinds.items()] or [["(none)", "0"]]
    )
    print_info(
        "This is the evaluation lab deliverable - "
        "each label with evidence you can show your mentor."
    )


# ------------------------------------------------------------------
# Main menu
# ------------------------------------------------------------------

def run(engine: RAGEngine = None):
    """Retrieval Lab - evaluation and inspection tools for the merged RAG pipeline."""
    print_feature_header("Retrieval Lab")

    print_concept(
        "The merged RAG",
        "The main chat (RAG option 3) now runs the improved\n"
        "pipeline: vector search + BM25 keyword search fused with\n"
        "RRF, then a cross-encoder rerank - not vector-only.\n\n"
        "This lab is the toolbox around it:\n"
        "  - Concepts: the two failure kinds + every technique\n"
        "  - Inspection: what was fetched, for any question\n"
        "  - Before/After: the measured numbers for your PR\n"
        "  - Labeling: evidence for each failure's kind\n\n"
        "Eval data: 18 labeled HR questions in "
        "data/eval_questions.json\n"
        "(edit them to add your own questions!)."
    )

    print_divider()

    # ✅ Engine + evaluator (builds BM25 lazily, loads eval set)
    if engine is None:
        engine = RAGEngine()

    try:
        evaluator = Evaluator()
    except FileNotFoundError as e:
        print_error(f"{e}")
        print_info("Expected file: data/eval_questions.json")
        return

    print_success(
        f"Eval set ready: {len(evaluator.questions)} questions "
        f"over {len(set(q['expected_source'] for q in evaluator.questions))} "
        f"documents"
    )

    while True:
        console.print(
            "\n[bold cyan]Retrieval Lab Options:[/bold cyan]\n"
            "  [green]1[/green] → Concepts (the two failures, "
            "hybrid, rerank...)\n"
            "  [green]2[/green] → Inspection view "
            "(question | fetched | answer)\n"
            "  [green]3[/green] → Before/After: baseline vs "
            "ONE change\n"
            "  [green]4[/green] → Compare ALL variants (explore)\n"
            "  [green]5[/green] → Label the failures\n"
            "  [green]0[/green] → Back to RAG menu\n"
        )

        choice = get_user_input("Choose option: ")

        if choice == "0":
            break
        elif choice == "1":
            print_divider()
            show_concepts()
        elif choice == "2":
            print_divider()
            inspection_view(engine, evaluator)
        elif choice == "3":
            print_divider()
            run_before_after(engine, evaluator)
        elif choice == "4":
            print_divider()
            compare_all_variants(engine, evaluator)
        elif choice == "5":
            print_divider()
            label_failures(engine, evaluator)
        else:
            print_error("Invalid option!")
