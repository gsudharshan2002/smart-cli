from src.ai_client import ask_ai
from src.utils.printer import (
    print_feature_header,
    print_concept,
    print_prompt,
    print_response,
    print_step,
    print_thinking,
    print_info,
    print_divider
)
from src.utils.menu import get_user_input


def run():
    """Top-K & Top-P Feature"""

    print_feature_header("Top-K & Top-P Sampling")

    print_concept(
        "What is Top-K?",
        "Top-K limits the AI to only consider the\n"
        "TOP K most probable next words.\n\n"
        "• Top-K = 1   → Always picks most likely word (robotic)\n"
        "• Top-K = 10  → Picks from top 10 words (balanced)\n"
        "• Top-K = 100 → Picks from top 100 words (creative)\n\n"
        "Lower K = More focused\n"
        "Higher K = More diverse"
    )

    print_concept(
        "What is Top-P (Nucleus Sampling)?",
        "Top-P picks words until their combined\n"
        "probability reaches P.\n\n"
        "• Top-P = 0.1 → Very focused, only top words\n"
        "• Top-P = 0.5 → Balanced selection\n"
        "• Top-P = 0.9 → Wide selection, more creative\n"
        "• Top-P = 1.0 → All words considered\n\n"
        "Lower P = More focused\n"
        "Higher P = More diverse"
    )

    prompt = get_user_input(
        "📝 Enter a prompt (e.g. 'Tell me a fun fact about space'): "
    )

    if not prompt:
        prompt = "Tell me a fun fact about space"
        print_info(f"Using default: {prompt}")

    print_divider()

    print_step(
        "Testing Top-P",
        "Same prompt at different Top-P values..."
    )

    top_p_values = [0.1, 0.5, 0.9]

    for top_p in top_p_values:
        print_step(
            f"Top-P = {top_p}",
            "Sending to AI..."
        )
        print_thinking()

        response = ask_ai(
            prompt=prompt,
            top_p=top_p,
            temperature=0.7
        )

        print_response(
            f"[Top-P: {top_p}]\n\n{response}"
        )

    print_divider()

    print_info(
        "Notice:\n"
        "• Low Top-P  = Safe, predictable responses\n"
        "• High Top-P = Creative, diverse responses\n"
        "• Combine with Temperature for best results!"
    )

    print_divider()

    print_step(
        "Bonus Test",
        "Combining Top-P + Temperature together..."
    )

    combos = [
        {"temp": 0.2, "top_p": 0.1, "label": "Very Focused"},
        {"temp": 0.7, "top_p": 0.5, "label": "Balanced"},
        {"temp": 1.2, "top_p": 0.9, "label": "Very Creative"},
    ]

    for combo in combos:
        print_step(
            combo["label"],
            f"Temp={combo['temp']} | Top-P={combo['top_p']}"
        )
        print_thinking()

        response = ask_ai(
            prompt=prompt,
            temperature=combo["temp"],
            top_p=combo["top_p"]
        )

        print_response(
            f"[{combo['label']}]\n"
            f"Temperature: {combo['temp']} | "
            f"Top-P: {combo['top_p']}\n\n"
            f"{response}"
        )

    print_info(
        "Best Practice:\n"
        "• Use Temperature OR Top-P not both high\n"
        "• For facts  → Low Temp + Low Top-P\n"
        "• For stories → High Temp + High Top-P"
    )