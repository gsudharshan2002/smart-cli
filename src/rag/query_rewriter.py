# src/rag/query_rewriter.py - Query rewriting & HyDE

from src.ai_client import ask_ai


class QueryRewriter:
    """
    Two ways to fix a bad user question BEFORE search:

    1. Query rewriting (cheap, one LLM call)
       "whats the max they give me back if my hotel is pricey while
       im in another country for work" 
         → "maximum hotel reimbursement rate international travel"

    2. HyDE (Hypothetical Document Embeddings)
       Instead of rewriting the question, ask the LLM to write the
       ANSWER as it would appear in a policy document, then embed
       THAT. The theory: your question text and the policy text
       live in different vocabularies, but a hypothetical answer
       and the real policy paragraph share one.

       Retrieval then looks for "sentences that answer this
       question" instead of "sentences that look like this
       question".
    """

    def rewrite(
        self,
        question: str,
        hypothetically: bool = False,
        temperature: float = 0.0
    ) -> str:
        """
        Returns the improved search query string.

        hypothetically=True -> HyDE: write a fake policy answer.
        hypothetically=False -> plain rewrite into a clean query.
        """

        if hypothetically:
            prompt = f"""
Original user question: {question}

Write a 3-4 sentence hypothetical ANSWER to this question, the way
it would appear in a real company policy document. Be specific and
include exact numbers, names, and codes where they would appear.

Do NOT answer as yourself. Write it as policy text:
"""
        else:
            prompt = f"""
Original user question: {question}

Rewrite this into ONE clean search query that will find the exact
paragraph in a policy document that answers it.

Rules:
- Keep exact numbers, codes, and proper nouns (e.g. "$325", "10
  days", "ERR-4032", "workers compensation") UNCHANGED
- Remove filler words and conversational phrasing
- Max 12 words
- Output ONLY the rewritten query, no quotes, no explanation
"""

        return ask_ai(
            prompt=prompt,
            system=(
                "You are a search-query optimizer for document "
                "retrieval. Output only the query, nothing else."
            ),
            temperature=temperature,
            max_tokens=1024
        ).strip().strip('"') or question
