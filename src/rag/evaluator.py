# src/rag/evaluator.py - Retrieval evaluation (hit-rate@k, recall@k, MRR)

import json
import os

from src.rag.hybrid import SmartRetriever
from src.rag.config import DATA_PATH

EVAL_PATH = os.path.join(DATA_PATH, "eval_questions.json")
LABELS_PATH = os.path.join(DATA_PATH, "eval_labels.json")


class Evaluator:
    """
    Measures whether the RIGHT DOCUMENT shows up, with a number.

    The Week 5 mantra: before you touch anything, measure. After
    one change, measure again. If the number didn't move, the
    change didn't help.

    Metrics (k = top-k retrieved chunks):
      hit_rate@k : fraction of questions where the expected DOCUMENT
                   appears anywhere in the top-k chunks.
                   This is the number the course cares about.
      recall@k   : fraction of questions where the exact expected
                   CHUNK (source + phrase) appears in top-k.
      mrr@k      : Mean Reciprocal Rank - average of 1/rank of the
                   first expected-document chunk. Rewards being at
                   #1 instead of just being on the list.
    """

    # variant -> (use_bm25, use_rerank, use_rewrite, use_hyde)
    VARIANTS = {
        "vector":            (False, False, False, False),
        "hybrid":            (True,  False, False, False),
        "hybrid+rerank":     (True,  True,  False, False),
        "rewrite":           (True,  True,  True,  False),
        "hyde":              (True,  True,  False, True),
    }

    def __init__(self, questions_path: str = EVAL_PATH):
        self.questions_path = questions_path
        self.questions = self._load_questions()
        self.retriever = SmartRetriever()
        self.ground_truth = {}   # question_id -> chunk id
        self._resolve_ground_truth()

    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------

    def _load_questions(self) -> list:
        if not os.path.exists(self.questions_path):
            raise FileNotFoundError(
                f"Eval file not found: {self.questions_path}"
            )
        with open(self.questions_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["questions"]

    def _resolve_ground_truth(self):
        """
        Find the exact chunk that answers each question.

        A question is correctly answered when the expected chunk is
        retrieved (chunk-level) or when ANY chunk from the expected
        document is retrieved (document-level, the course's metric).
        """
        db = self.retriever.db
        collection = db.get_collection()
        data = collection.get(
            include=["documents", "metadatas"]
        )

        warnings = []

        for q in self.questions:
            qid = q["id"]
            source = q["expected_source"]
            phrase = q.get("expected_phrase", "").lower()

            candidates = [
                data["ids"][i]
                for i in range(len(data["ids"]))
                if data["metadatas"][i].get("source") == source
            ]

            if not candidates:
                warnings.append(
                    f"Q{qid}: no chunks found for source "
                    f"'{source}' (document not indexed?)"
                )
                self.ground_truth[qid] = None
                continue

            # Normalize whitespace: PDF extraction often leaves
            # double spaces inside phrases ('covered by  workers').
            def norm(s: str) -> str:
                return " ".join(s.lower().split())

            exact = None
            if phrase:
                for i in range(len(data["ids"])):
                    if (
                        data["metadatas"][i].get("source") == source
                        and norm(phrase) in norm(data["documents"][i])
                    ):
                        exact = data["ids"][i]
                        break

            if exact is None and phrase:
                warnings.append(
                    f"Q{qid}: phrase '{phrase}' not found in any "
                    f"chunk of '{source}' - falling back to "
                    f"document-level check only"
                )

            self.ground_truth[qid] = exact or (candidates[0] if candidates else None)

        if warnings:
            print("\n    ⚠️  Ground-truth warnings:")
            for w in warnings:
                print(f"      - {w}")
            print()

    # ------------------------------------------------------------
    # Running one variant
    # ------------------------------------------------------------

    def evaluate_variant(
        self,
        variant: str,
        top_k: int = 3,
        verbose: bool = False
    ) -> dict:
        """Run the eval set through ONE retrieval configuration."""
        if variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Options: {list(self.VARIANTS)}"
            )

        use_bm25, use_rerank, use_rewrite, use_hyde = (
            self.VARIANTS[variant]
        )

        print(f"\n    🔄 Running eval variant: '{variant}' "
              f"(top_k={top_k})")

        rows = []

        for q in self.questions:
            qid = q["id"]

            result = self.retriever.retrieve(
                query=q["question"],
                top_k=top_k,
                use_bm25=use_bm25,
                use_rerank=use_rerank,
                use_rewrite=use_rewrite,
                use_hyde=use_hyde
            )

            chunks = result["chunks"]

            # ✅ Compute this question's 3 numbers
            expected_source = q["expected_source"]
            expected_chunk = self.ground_truth.get(qid)

            hit = any(
                c["metadata"].get("source") == expected_source
                for c in chunks
            )
            recalled = any(c["id"] == expected_chunk for c in chunks)

            rr = 0.0
            for rank, c in enumerate(chunks, 1):
                if c["metadata"].get("source") == expected_source:
                    rr = 1.0 / rank
                    break

            rows.append({
                "id": qid,
                "question": q["question"],
                "expected_source": expected_source,
                "retrieved_sources": [
                    c["metadata"].get("source", "?")
                    for c in chunks
                ],
                "hit": hit,
                "recall": recalled,
                "rr": rr,
                "expected_chunk_retrieved": recalled,
                "search_query": result.get("search_query", q["question"]),
            })

            if verbose:
                mark = "✅" if hit else "❌"
                print(f"      {mark} Q{qid}: {q['question'][:60]}")

        return {
            "variant": variant,
            "top_k": top_k,
            "rows": rows,
            "metrics": self._compute_metrics(rows),
        }

    def _compute_metrics(self, rows: list) -> dict:
        n = len(rows)
        return {
            "hit_rate": round(
                sum(1 for r in rows if r["hit"]) / n, 4
            ) if n else 0,
            "recall": round(
                sum(1 for r in rows if r["recall"]) / n, 4
            ) if n else 0,
            "mrr": round(
                sum(r["rr"] for r in rows) / n, 4
            ) if n else 0,
            "questions": n,
            "hits": sum(1 for r in rows if r["hit"]),
        }

    # ------------------------------------------------------------
    # Before / after comparison
    # ------------------------------------------------------------

    def compare(
        self,
        baseline: str = "vector",
        improved: str = "hybrid",
        top_k: int = 3,
        verbose: bool = True
    ) -> dict:
        """
        Run baseline + one improvement, return a before/after
        comparison - the deliverable the course asks for.
        """
        before = self.evaluate_variant(baseline, top_k, verbose=verbose)
        after = self.evaluate_variant(improved, top_k, verbose=verbose)

        b = before["metrics"]
        a = after["metrics"]

        comparison = {
            "baseline": baseline,
            "improved": improved,
            "top_k": top_k,
            "metrics": {
                "hit_rate": {
                    "before": b["hit_rate"],
                    "after": a["hit_rate"],
                    "delta": round(a["hit_rate"] - b["hit_rate"], 4),
                },
                "recall": {
                    "before": b["recall"],
                    "after": a["recall"],
                    "delta": round(a["recall"] - b["recall"], 4),
                },
                "mrr": {
                    "before": b["mrr"],
                    "after": a["mrr"],
                    "delta": round(a["mrr"] - b["mrr"], 4),
                },
            },
            "per_question": self._diff_rows(
                before["rows"], after["rows"]
            ),
        }
        return comparison

    def _diff_rows(self, before_rows: list, after_rows: list) -> list:
        before = {r["id"]: r for r in before_rows}
        after = {r["id"]: r for r in after_rows}

        out = []
        for qid in sorted(set(before) | set(after)):
            b = before.get(qid)
            a = after.get(qid)

            if b["hit"] and a["hit"]:
                status = "✅ both"
            elif b["hit"] and not a["hit"]:
                status = "⬇️  regression!"
            elif not b["hit"] and a["hit"]:
                status = "🎉 fixed!"
            else:
                status = "❌ both fail"

            out.append({
                "id": qid,
                "question": a["question"],
                "expected_source": a["expected_source"],
                "before_hit": b["hit"],
                "after_hit": a["hit"],
                "status": status,
                "after_sources": a["retrieved_sources"],
            })

        return out

    # ------------------------------------------------------------
    # Failure classification ("label the failures")
    # ------------------------------------------------------------

    def load_labels(self) -> dict:
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_label(
        self,
        question_id: int,
        failure_kind: str,
        note: str = ""
    ):
        labels = self.load_labels()
        labels[str(question_id)] = {
            "failure_kind": failure_kind,
            "note": note,
        }
        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2, ensure_ascii=False)

    def get_question(self, question_id: int) -> dict:
        for q in self.questions:
            if q["id"] == question_id:
                return q
        return None

    def measure_latency(self, top_k: int = 3) -> dict:
        """
        Measure p50 latency before and after (vector-only vs hybrid+rerank).

        Returns dict with p50 latencies and delta in milliseconds.
        Used for Evaluation Rubric Criterion 3.
        """
        import time
        import numpy as np

        # Measure vector-only latency
        vector_durations = []
        for q in self.questions:
            start = time.time()
            self.retriever.retrieve(
                q["question"], top_k=top_k, use_bm25=False, use_rerank=False
            )
            vector_durations.append(time.time() - start)

        # Measure hybrid+rerank latency
        rerank_durations = []
        for q in self.questions:
            start = time.time()
            self.retriever.retrieve(
                q["question"], top_k=top_k, use_bm25=True, use_rerank=True
            )
            rerank_durations.append(time.time() - start)

        # Calculate p50 (median) latency
        vector_p50 = np.median(vector_durations) * 1000  # convert to ms
        rerank_p50 = np.median(rerank_durations) * 1000  # convert to ms
        latency_delta = rerank_p50 - vector_p50

        return {
            "vector_p50_ms": round(vector_p50, 1),
            "rerank_p50_ms": round(rerank_p50, 1),
            "delta_ms": round(latency_delta, 1),
            "questions": len(self.questions),
        }
