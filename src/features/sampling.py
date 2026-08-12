from src.ai_client import ask_ai
from src.utils.printer import (
    print_feature_header,
    print_concept,
    print_prompt,
    print_response,
    print_step,
    print_thinking,
    print_table,
    print_info
)
from src.utils.menu import get_user_input


def run():
    """Sampling & Temperature Feature"""

    print_feature_header("Sampling & Temperature")

    print_concept(
        "What is Temperature?",
        "Temperature controls the randomness of AI responses.\n\n"
        "• LOW (0.0 - 0.3)  → Focused, predictable, factual\n"
        "• MEDIUM (0.4 - 0.7) → Balanced creativity\n"
        "• HIGH (0.8 - 2.0)  → Creative, random, surprising\n\n"
        "We will ask the SAME prompt at different temperatures\n"
        "so you can SEE the difference!"
    )

    prompt = get_user_input(
        "📝 Enter a creative prompt (e.g. 'Write a tagline for coffee'): "
    )

    if not prompt:
        prompt = "Write a creative tagline for a coffee shop"
        print_info(f"Using default prompt: {prompt}")

    temperatures = [0.0, 0.7, 1.5]

    print_step("Testing", "Same prompt at 3 different temperatures...")
    print()

    results = []

    for temp in temperatures:
        print_step(f"Temperature {temp}", "Sending to AI...")
        print_thinking()

        response = ask_ai(prompt, temperature=temp)
        results.append([str(temp), response])

        print_response(f"[Temp {temp}]\n\n{response}")

    print_info("Notice how higher temperature = more creative/random!")