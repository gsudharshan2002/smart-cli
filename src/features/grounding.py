from src.ai_client import ask_ai
from src.utils.printer import (
    print_feature_header,
    print_concept,
    print_response,
    print_step,
    print_thinking,
    print_info,
    print_divider,
    print_prompt
)
from src.utils.menu import get_user_input


# ✅ Fake company data (grounding context)
COMPANY_CONTEXT = """
Company Name: TechNova Inc
Founded: 2019
CEO: Sarah Johnson
Products:
  - NovaCloud (Cloud Storage)
  - NovaChat (Team Messaging)
  - NovaAI (AI Assistant)
Employees: 450
HQ: San Francisco, CA
Revenue 2023: $12M
Mission: Making AI accessible to everyone
"""


def run():
    """Grounding Feature"""

    print_feature_header("Grounding")

    print_concept(
        "What is Grounding?",
        "Grounding means giving the AI real context/facts\n"
        "so it answers based on YOUR data.\n\n"
        "Without Grounding:\n"
        "  ❌ AI uses only training data\n"
        "  ❌ AI may hallucinate wrong facts\n"
        "  ❌ AI doesn't know your specific info\n\n"
        "With Grounding:\n"
        "  ✅ AI uses YOUR provided facts\n"
        "  ✅ Accurate & relevant answers\n"
        "  ✅ AI knows your specific context\n\n"
        "Used in: RAG, Chatbots, Customer Support"
    )

    print_divider()

    print_step(
        "Demo 1",
        "Asking AI WITHOUT grounding (no context)"
    )

    question = "What products does TechNova Inc offer?"

    print_prompt(
        f"Question: {question}\n"
        f"Context: None (no grounding)"
    )
    print_thinking()

    response_without = ask_ai(
        prompt=question,
        system="You are a helpful assistant."
    )

    print_response(
        f"[WITHOUT Grounding]\n\n{response_without}"
    )

    print_info(
        "AI doesn't know TechNova!\n"
        "It either guessed or said it doesn't know."
    )

    print_divider()

    print_step(
        "Demo 2",
        "Asking AI WITH grounding (real context provided)"
    )

    grounded_system = f"""
You are a helpful assistant for TechNova Inc.
Use ONLY the following information to answer questions.
If the answer is not in the context, say 'I don't have that information'.

COMPANY INFORMATION:
{COMPANY_CONTEXT}
"""

    print_prompt(
        f"Question: {question}\n"
        f"Context: TechNova company data provided ✅"
    )
    print_thinking()

    response_with = ask_ai(
        prompt=question,
        system=grounded_system
    )

    print_response(
        f"[WITH Grounding]\n\n{response_with}"
    )

    print_info(
        "Now AI knows exactly about TechNova!\n"
        "It answered from YOUR provided context."
    )

    print_divider()

    print_step(
        "Demo 3",
        "Try your own question with grounding!"
    )

    print_info(
        "Available context about TechNova:\n"
        f"{COMPANY_CONTEXT}"
    )

    custom_question = get_user_input(
        "📝 Ask anything about TechNova: "
    )

    if not custom_question:
        custom_question = "Who is the CEO of TechNova?"
        print_info(f"Using default: {custom_question}")

    print_prompt(
        f"Question: {custom_question}\n"
        f"Context: TechNova data grounded ✅"
    )
    print_thinking()

    custom_response = ask_ai(
        prompt=custom_question,
        system=grounded_system
    )

    print_response(
        f"[Grounded Answer]\n\n{custom_response}"
    )

    print_divider()

    print_step(
        "Demo 4",
        "Hallucination Test — Ask something NOT in context"
    )

    unknown_question = "What is TechNova's office address in New York?"

    print_prompt(
        f"Question: {unknown_question}\n"
        f"Note: This info is NOT in our context!"
    )
    print_thinking()

    hallucination_response = ask_ai(
        prompt=unknown_question,
        system=grounded_system
    )

    print_response(
        f"[Grounded AI - Unknown Info]\n\n"
        f"{hallucination_response}"
    )

    print_info(
        "With proper grounding:\n"
        "AI says 'I don't have that info'\n"
        "✅ Instead of making up wrong answers\n"
        "✅ This is how RAG systems work!"
    )

    print_divider()

    print_concept(
        "Grounding Summary",
        "Key Takeaways:\n\n"
        "1. Always provide context for specific questions\n"
        "2. Tell AI to ONLY use provided context\n"
        "3. Grounding prevents hallucination\n"
        "4. Used in RAG, Chatbots, Search systems\n\n"
        "Real World Uses:\n"
        "  • Customer support bots\n"
        "  • Document Q&A systems\n"
        "  • Company knowledge bases\n"
        "  • Medical/Legal AI assistants"
    )