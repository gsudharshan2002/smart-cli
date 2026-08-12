# src/features/self_consistency.py

from src.ai_client import ask_ai
from src.utils.printer import (
    print_feature_header,
    print_concept,
    print_response,
    print_step,
    print_thinking,
    print_info,
    print_divider,
    print_prompt,
    print_success
)
from src.utils.menu import get_user_input


def get_multiple_responses(
    prompt: str,
    system: str,
    times: int = 3,
    temperature: float = 0.7
) -> list:
    """
    Ask AI same question multiple times
    Returns list of responses
    """
    responses = []

    for i in range(times):
        print_step(
            f"Run {i + 1}/{times}",
            "Asking AI same question..."
        )
        print_thinking()

        response = ask_ai(
            prompt=prompt,
            system=system,
            temperature=temperature
        )

        responses.append(response)

        print_response(
            f"[Answer {i + 1}]\n\n{response}"
        )

    return responses


def find_consistent_answer(
    responses: list,
    prompt: str
) -> str:
    """
    Ask AI to analyze responses
    and find most consistent answer
    """

    analysis_prompt = f"""
Original Question:
{prompt}

Here are {len(responses)} different answers to the same question:

{chr(10).join([f"Answer {i+1}: {r}" for i, r in enumerate(responses)])}

Analyze all answers above and:
1. Find what is CONSISTENT across all answers
2. Find what is DIFFERENT between answers
3. Give the BEST final answer based on consistency
4. Explain why this is the most reliable answer

Format your response as:
CONSISTENT POINTS: ...
DIFFERENCES: ...
BEST ANSWER: ...
WHY RELIABLE: ...
"""

    return ask_ai(
        prompt=analysis_prompt,
        system="You are an expert at analyzing and comparing answers for consistency.",
        temperature=0.1
    )


def run():
    """Self Consistency Feature"""

    # ✅ Header
    print_feature_header("Self Consistency")

    # ✅ Explain concept
    print_concept(
        "What is Self Consistency?",
        "Self Consistency = Ask AI same question MULTIPLE times\n"
        "then find the most consistent answer.\n\n"
        "Why it works:\n"
        "  • Single AI response can be wrong/biased\n"
        "  • Multiple responses show patterns\n"
        "  • Consistent answers = more reliable\n"
        "  • Different answers = uncertain topic\n\n"
        "Process:\n"
        "  Step 1 → Ask same question N times\n"
        "  Step 2 → Collect all responses\n"
        "  Step 3 → Find common patterns\n"
        "  Step 4 → Pick most consistent answer\n\n"
        "Used in: Fact checking, Math, Reasoning tasks"
    )

    print_divider()

    # ✅ Demo 1: Simple Fact Question
    print_step(
        "Demo 1",
        "Fact Question — Ask 3 times & check consistency"
    )

    fact_question = (
        "What are the 3 most important skills "
        "for a software developer?"
    )

    print_prompt(
        f"Question: {fact_question}\n"
        f"Will ask: 3 times\n"
        f"Temperature: 0.7 (some variation)"
    )

    print_divider()

    # ✅ Get 3 responses
    responses = get_multiple_responses(
        prompt=fact_question,
        system="You are an expert software engineering mentor.",
        times=3,
        temperature=0.7
    )

    print_divider()

    # ✅ Analyze consistency
    print_step(
        "Analysis",
        "Finding most consistent answer..."
    )
    print_thinking()

    consistent_answer = find_consistent_answer(
        responses=responses,
        prompt=fact_question
    )

    print_response(
        f"[Consistency Analysis]\n\n{consistent_answer}"
    )

    print_success("Self consistency analysis complete!")

    print_divider()

    # ✅ Demo 2: Reasoning Question
    print_step(
        "Demo 2",
        "Reasoning Question — Self consistency with logic"
    )

    reasoning_question = (
        "A store sells apples for $2 each and oranges for $3 each. "
        "If someone buys 4 apples and 3 oranges, "
        "how much do they spend in total? "
        "Show your reasoning step by step."
    )

    print_prompt(
        f"Question: {reasoning_question}\n"
        f"Will ask: 3 times\n"
        f"Temperature: 0.3 (less variation for math)"
    )

    print_divider()

    math_responses = get_multiple_responses(
        prompt=reasoning_question,
        system=(
            "You are a math tutor. "
            "Always show step by step reasoning."
        ),
        times=3,
        temperature=0.3
    )

    print_divider()

    print_step(
        "Analysis",
        "Checking if all math answers are consistent..."
    )
    print_thinking()

    math_analysis = find_consistent_answer(
        responses=math_responses,
        prompt=reasoning_question
    )

    print_response(
        f"[Math Consistency Analysis]\n\n{math_analysis}"
    )

    print_info(
        "For math questions:\n"
        "✅ All answers should be same (17)\n"
        "✅ If different → AI made reasoning error\n"
        "✅ Consistency check catches mistakes!"
    )

    print_divider()

    # ✅ Demo 3: Custom question
    print_step(
        "Demo 3",
        "Try YOUR own question!"
    )

    custom_question = get_user_input(
        "📝 Enter your question to test consistency: "
    )

    if not custom_question:
        custom_question = "What is the best programming language to learn first?"
        print_info(f"Using default: {custom_question}")

    times_input = get_user_input(
        "🔢 How many times to ask? (2-5, default 3): "
    )

    try:
        times = int(times_input)
        if times < 2 or times > 5:
            times = 3
    except ValueError:
        times = 3

    print_info(f"Asking AI {times} times...")
    print_divider()

    custom_responses = get_multiple_responses(
        prompt=custom_question,
        system="You are a helpful and knowledgeable assistant.",
        times=times,
        temperature=0.7
    )

    print_divider()

    print_step(
        "Final Analysis",
        f"Analyzing {times} responses for consistency..."
    )
    print_thinking()

    custom_analysis = find_consistent_answer(
        responses=custom_responses,
        prompt=custom_question
    )

    print_response(
        f"[Your Question Consistency Analysis]\n\n"
        f"{custom_analysis}"
    )

    print_divider()

    # ✅ Summary
    print_concept(
        "Self Consistency Summary",
        "Key Takeaways:\n\n"
        "1. Single response can be unreliable\n"
        "2. Multiple responses reveal patterns\n"
        "3. Consistent = more trustworthy\n"
        "4. Different = topic is uncertain\n\n"
        "Best used for:\n"
        "  ✅ Math & logic problems\n"
        "  ✅ Fact verification\n"
        "  ✅ Important decisions\n"
        "  ✅ Complex reasoning tasks\n\n"
        "Temperature Tips:\n"
        "  • Math → Low temp (0.1-0.3)\n"
        "  • Creative → High temp (0.7-1.0)\n"
        "  • Facts → Medium temp (0.4-0.6)"
    )