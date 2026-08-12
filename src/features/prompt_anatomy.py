# src/features/prompt_anatomy.py

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


def build_prompt(
    role: str,
    context: str,
    task: str,
    format: str,
    constraints: str
) -> str:
    """Build a structured prompt from parts"""
    return f"""
ROLE:
{role}

CONTEXT:
{context}

TASK:
{task}

OUTPUT FORMAT:
{format}

CONSTRAINTS:
{constraints}
""".strip()


def run():
    """Prompt Anatomy Feature"""

    # ✅ Header
    print_feature_header("Prompt Anatomy")

    # ✅ Explain concept
    print_concept(
        "What is Prompt Anatomy?",
        "Prompt Anatomy is the structure of a well-built prompt.\n\n"
        "A perfect prompt has 5 key parts:\n\n"
        "1. 🎭 ROLE        → Who the AI should be\n"
        "2. 📖 CONTEXT     → Background information\n"
        "3. 📋 TASK        → What AI should do\n"
        "4. 📝 FORMAT      → How to structure output\n"
        "5. 🚫 CONSTRAINTS → Rules & limitations\n\n"
        "Better structure = Better AI responses!"
    )

    print_divider()

    print_step(
        "Demo 1",
        "BAD prompt — vague & unstructured"
    )

    bad_prompt = "Write about dogs"

    print_prompt(
        f"BAD Prompt:\n{bad_prompt}\n\n"
        f"Problems:\n"
        f"❌ No role defined\n"
        f"❌ No context given\n"
        f"❌ No format specified\n"
        f"❌ No constraints set"
    )
    print_thinking()

    bad_response = ask_ai(
        prompt=bad_prompt,
        system="You are a helpful assistant."
    )

    print_response(
        f"[BAD Prompt Response]\n\n{bad_response}"
    )

    print_info(
        "Response is random & unpredictable!\n"
        "AI doesn't know what you really want."
    )

    print_divider()

    # ✅ Demo 2: Good Prompt with 