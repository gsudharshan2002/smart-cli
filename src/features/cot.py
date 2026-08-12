# src/features/cot.py

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
    print_success,
    print_table
)
from src.utils.menu import get_user_input


def without_cot(question: str) -> str:
    """Ask AI directly without COT"""
    return ask_ai(
        prompt=question,
        system="You are a helpful assistant.",
        temperature=0.3
    )


def zero_shot_cot(question: str) -> str:
    """
    Zero Shot COT
    Just add 'think step by step'
    """
    prompt = (
        f"{question}\n\n"
        f"Think step by step before answering."
    )

    return ask_ai(
        prompt=prompt,
        system=(
            "You are a careful thinker who always "
            "reasons through problems step by step."
        ),
        temperature=0.3
    )


def manual_cot(question: str) -> str:
    """
    Manual COT
    Show AI examples of step by step thinking
    """
    prompt = f"""
Learn from these examples of step by step thinking:

Example 1:
Question: If a shop has 50 apples and sells 23, 
then receives 30 more, how many apples are there?
Thinking:
  Step 1: Start with 50 apples
  Step 2: Sell 23 → 50 - 23 = 27 apples left
  Step 3: Receive 30 more → 27 + 30 = 57 apples
Answer: 57 apples

Example 2:
Question: A train travels 60km/h for 2.5 hours. 
How far does it travel?
Thinking:
  Step 1: Speed = 60 km/h
  Step 2: Time = 2.5 hours
  Step 3: Distance = Speed × Time
  Step 4: Distance = 60 × 2.5 = 150 km
Answer: 150 km

Now solve this using the SAME thinking format:
Question: {question}
Thinking:
"""

    return ask_ai(
        prompt=prompt,
        system=(
            "You are a careful step by step problem solver. "
            "Always show your complete thinking process."
        ),
        temperature=0.2
    )


def auto_cot(question: str) -> str:
    """
    Auto COT
    AI generates its own reasoning chain
    """
    # ✅ Step 1: Generate reasoning steps
    steps_prompt = (
        f"For this question, list the key reasoning "
        f"steps needed to solve it correctly.\n\n"
        f"Question: {question}\n\n"
        f"List steps only, no answers yet:"
    )

    reasoning_steps = ask_ai(
        prompt=steps_prompt,
        system="You are an expert at breaking down problems.",
        temperature=0.3
    )

    # ✅ Step 2: Use steps to answer
    answer_prompt = (
        f"Question: {question}\n\n"
        f"Use these reasoning steps:\n"
        f"{reasoning_steps}\n\n"
        f"Now give the complete solution following "
        f"these steps carefully:"
    )

    final_answer = ask_ai(
        prompt=answer_prompt,
        system=(
            "You are a careful problem solver who "
            "follows reasoning steps precisely."
        ),
        temperature=0.2
    )

    return (
        f"REASONING STEPS:\n{reasoning_steps}\n\n"
        f"SOLUTION:\n{final_answer}"
    )


def self_ask_cot(question: str) -> str:
    """
    Self Ask COT
    AI asks itself sub-questions to solve problem
    """
    prompt = f"""
To answer this question, first ask yourself 
smaller sub-questions and answer them one by one.

Question: {question}

Format your response like this:
Sub-question 1: ...
Answer 1: ...

Sub-question 2: ...
Answer 2: ...

Sub-question 3: ...
Answer 3: ...

Final Answer: (combine all sub-answers)
"""

    return ask_ai(
        prompt=prompt,
        system=(
            "You are an expert who breaks questions "
            "into smaller sub-questions to solve them."
        ),
        temperature=0.3
    )


def reflective_cot(question: str) -> str:
    """
    Reflective COT
    AI answers then reflects and improves
    """

    # ✅ Step 1: Initial answer
    initial_answer = ask_ai(
        prompt=question,
        system="You are a helpful assistant.",
        temperature=0.5
    )

    # ✅ Step 2: Reflect on answer
    reflection_prompt = f"""
Question: {question}

Initial Answer:
{initial_answer}

Now reflect on this answer:
1. What is CORRECT in this answer?
2. What might be WRONG or MISSING?
3. What can be IMPROVED?
4. Give a BETTER final answer.
"""

    reflection = ask_ai(
        prompt=reflection_prompt,
        system=(
            "You are a critical thinker who "
            "always improves initial answers."
        ),
        temperature=0.3
    )

    return (
        f"INITIAL ANSWER:\n{initial_answer}\n\n"
        f"REFLECTION & IMPROVEMENT:\n{reflection}"
    )


def run():
    """Chain of Thoughts Feature"""

    # ✅ Header
    print_feature_header("Chain of Thoughts (COT)")

    # ✅ Explain concept
    print_concept(
        "What is Chain of Thoughts?",
        "COT = Force AI to THINK step by step\n"
        "before giving the final answer.\n\n"
        "Without COT:\n"
        "  Question → Direct Answer\n"
        "  (Fast but often wrong)\n\n"
        "With COT:\n"
        "  Question → Think → Think → Think\n"
        "  → Much better answer!\n\n"
        "COT Types we will demo:\n"
        "  1. Without COT    → Direct answer\n"
        "  2. Zero Shot COT  → 'Think step by step'\n"
        "  3. Manual COT     → Show examples\n"
        "  4. Auto COT       → AI generates steps\n"
        "  5. Self Ask COT   → AI asks sub-questions\n"
        "  6. Reflective COT → AI reflects & improves"
    )

    print_divider()

    # ✅ Demo 1: Math Problem
    print_step(
        "Demo 1",
        "Math Problem — Without vs With COT"
    )

    math_question = (
        "A store sells shirts for $25 each. "
        "They have a buy 2 get 1 free offer. "
        "If someone buys 6 shirts, "
        "how much do they pay in total?"
    )

    print_prompt(f"Math Question:\n{math_question}")

    print_divider()

    # Without COT
    print_step("Without COT", "Direct answer...")
    print_thinking()

    no_cot = without_cot(math_question)
    print_response(f"[Without COT]\n\n{no_cot}")

    print_divider()

    # Zero Shot COT
    print_step(
        "Zero Shot COT",
        "Adding 'think step by step'..."
    )

    print_info(
        "Magic phrase added:\n"
        "'Think step by step before answering'"
    )

    print_thinking()

    zero_cot = zero_shot_cot(math_question)
    print_response(f"[Zero Shot COT]\n\n{zero_cot}")

    print_divider()

    # Manual COT
    print_step(
        "Manual COT",
        "Showing thinking examples to AI..."
    )

    print_info(
        "Providing 2 examples of step by step thinking\n"
        "so AI learns the exact format we want"
    )

    print_thinking()

    manual = manual_cot(math_question)
    print_response(f"[Manual COT]\n\n{manual}")

    print_divider()

    # ✅ Demo 2: Logic Puzzle
    print_step(
        "Demo 2",
        "Logic Puzzle — Auto COT vs Self Ask COT"
    )

    logic_question = (
        "Three friends Alice, Bob, and Charlie "
        "each have different pets: a dog, cat, and bird. "
        "Alice does not have the dog. "
        "Bob does not have the cat. "
        "Charlie does not have the bird. "
        "Alice does not have the cat. "
        "Who has which pet?"
    )

    print_prompt(f"Logic Puzzle:\n{logic_question}")

    print_divider()

    # Auto COT
    print_step(
        "Auto COT",
        "AI generates its own reasoning steps..."
    )

    print_info(
        "Process:\n"
        "  Step 1: AI lists reasoning steps needed\n"
        "  Step 2: AI follows those steps to answer"
    )

    print_thinking()

    auto = auto_cot(logic_question)
    print_response(f"[Auto COT]\n\n{auto}")

    print_divider()

    # Self Ask COT
    print_step(
        "Self Ask COT",
        "AI asks itself sub-questions..."
    )

    print_info(
        "AI breaks the main question into\n"
        "smaller sub-questions and solves each one!"
    )

    print_thinking()

    self_ask = self_ask_cot(logic_question)
    print_response(f"[Self Ask COT]\n\n{self_ask}")

    print_divider()

    # ✅ Demo 3: Reflective COT
    print_step(
        "Demo 3",
        "Reflective COT — Answer then Improve"
    )

    reflect_question = (
        "What are the most important factors "
        "to consider when choosing a programming "
        "language for a new project?"
    )

    print_prompt(f"Question:\n{reflect_question}")

    print_info(
        "Process:\n"
        "  Step 1: AI gives initial answer\n"
        "  Step 2: AI reflects on its answer\n"
        "  Step 3: AI identifies gaps\n"
        "  Step 4: AI gives improved answer"
    )

    print_thinking()

    reflective = reflective_cot(reflect_question)
    print_response(f"[Reflective COT]\n\n{reflective}")

    print_divider()

    # ✅ Demo 4: COT comparison table
    print_step(
        "Demo 4",
        "COT Types Comparison"
    )

    print_table(
        title="🧠 COT Types Comparison",
        columns=["Type", "How", "Best For"],
        rows=[
            [
                "Without COT",
                "Direct question",
                "Simple facts"
            ],
            [
                "Zero Shot COT",
                "Add 'think step by step'",
                "Quick improvement"
            ],
            [
                "Manual COT",
                "Show examples",
                "Specific format needed"
            ],
            [
                "Auto COT",
                "AI generates steps",
                "Complex problems"
            ],
            [
                "Self Ask COT",
                "AI asks sub-questions",
                "Multi-part problems"
            ],
            [
                "Reflective COT",
                "Answer then improve",
                "Quality critical tasks"
            ],
        ]
    )

    print_divider()

    # ✅ Demo 5: Custom question
    print_step(
        "Demo 5",
        "Try YOUR own question with COT!"
    )

    print_table(
        title="Choose COT Type",
        columns=["Option", "Type"],
        rows=[
            ["1", "Zero Shot COT"],
            ["2", "Manual COT"],
            ["3", "Auto COT"],
            ["4", "Self Ask COT"],
            ["5", "Reflective COT"],
        ]
    )

    cot_choice = get_user_input(
        "Choose COT type (1-5): "
    )

    custom_question = get_user_input(
        "📝 Enter your question: "
    )

    if not custom_question:
        custom_question = (
            "Should I learn Python or JavaScript first "
            "as a beginner programmer?"
        )
        print_info(f"Using default: {custom_question}")

    print_prompt(f"Your Question:\n{custom_question}")
    print_thinking()

    if cot_choice == "1":
        result = zero_shot_cot(custom_question)
        cot_type = "Zero Shot COT"

    elif cot_choice == "2":
        result = manual_cot(custom_question)
        cot_type = "Manual COT"

    elif cot_choice == "3":
        result = auto_cot(custom_question)
        cot_type = "Auto COT"

    elif cot_choice == "4":
        result = self_ask_cot(custom_question)
        cot_type = "Self Ask COT"

    elif cot_choice == "5":
        result = reflective_cot(custom_question)
        cot_type = "Reflective COT"

    else:
        result = zero_shot_cot(custom_question)
        cot_type = "Zero Shot COT"

    print_response(f"[{cot_type}]\n\n{result}")

    print_divider()

    # ✅ Summary
    print_concept(
        "Chain of Thoughts Summary",
        "Key Takeaways:\n\n"
        "Core Idea:\n"
        "  Force AI to THINK before answering\n\n"
        "Simple trick:\n"
        "  Add 'Think step by step' to any prompt!\n"
        "  This alone improves accuracy by a lot!\n\n"
        "Best COT for each task:\n"
        "  Math problems  → Manual COT\n"
        "  Logic puzzles  → Self Ask COT\n"
        "  Complex topics → Auto COT\n"
        "  Writing tasks  → Reflective COT\n"
        "  Quick tasks    → Zero Shot COT\n\n"
        "Real World Uses:\n"
        "  ✅ Math & science problems\n"
        "  ✅ Code debugging\n"
        "  ✅ Legal & medical reasoning\n"
        "  ✅ Business decision making\n"
        "  ✅ Any complex problem solving"
    )