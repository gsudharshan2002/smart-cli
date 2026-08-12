# src/features/zero_few_shot.py

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


# ✅ Example datasets for demonstrations

SENTIMENT_EXAMPLES = [
    {"text": "I love this product!", "label": "POSITIVE"},
    {"text": "This is terrible.", "label": "NEGATIVE"},
    {"text": "It was okay I guess.", "label": "NEUTRAL"},
    {"text": "Best purchase ever!", "label": "POSITIVE"},
    {"text": "Complete waste of money.", "label": "NEGATIVE"},
]

TRANSLATION_EXAMPLES = [
    {"english": "Hello", "tamil": "வணக்கம்"},
    {"english": "Thank you", "tamil": "நன்றி"},
    {"english": "Good morning", "tamil": "காலை வணக்கம்"},
    {"english": "How are you?", "tamil": "நீங்கள் எப்படி இருக்கிறீர்கள்?"},
]

EMAIL_EXAMPLES = [
    {
        "tone": "Formal",
        "input": "Meeting tomorrow 3pm",
        "output": (
            "Dear Sir/Madam, I would like to inform you "
            "that our meeting is scheduled for tomorrow "
            "at 3:00 PM. Please confirm your availability."
        )
    },
    {
        "tone": "Casual",
        "input": "Meeting tomorrow 3pm",
        "output": (
            "Hey! Just a heads up, we have a meeting "
            "tomorrow at 3pm. See you there!"
        )
    },
]


def zero_shot_prompt(task: str, input_text: str) -> str:
    """Build zero shot prompt - no examples"""
    return f"{task}\n\nInput: {input_text}\nOutput:"


def one_shot_prompt(
    task: str,
    example: dict,
    input_text: str
) -> str:
    """Build one shot prompt - one example"""
    return (
        f"{task}\n\n"
        f"Example:\n"
        f"Input: {example['input']}\n"
        f"Output: {example['output']}\n\n"
        f"Now do the same:\n"
        f"Input: {input_text}\n"
        f"Output:"
    )


def few_shot_prompt(
    task: str,
    examples: list,
    input_text: str
) -> str:
    """Build few shot prompt - multiple examples"""
    examples_text = "\n\n".join([
        f"Example {i+1}:\n"
        f"Input: {ex['input']}\n"
        f"Output: {ex['output']}"
        for i, ex in enumerate(examples)
    ])

    return (
        f"{task}\n\n"
        f"{examples_text}\n\n"
        f"Now do the same:\n"
        f"Input: {input_text}\n"
        f"Output:"
    )


def sentiment_zero_shot(text: str) -> str:
    """Sentiment analysis with zero shot"""
    prompt = (
        f"Classify the sentiment of this text as "
        f"POSITIVE, NEGATIVE, or NEUTRAL.\n\n"
        f"Text: {text}\n"
        f"Sentiment:"
    )

    return ask_ai(
        prompt=prompt,
        system="You are a sentiment analysis expert.",
        temperature=0.1
    )


def sentiment_few_shot(text: str, examples: list) -> str:
    """Sentiment analysis with few shot examples"""

    examples_text = "\n".join([
        f"Text: {ex['text']} → {ex['label']}"
        for ex in examples
    ])

    prompt = (
        f"Classify sentiment as POSITIVE, NEGATIVE, or NEUTRAL.\n\n"
        f"Learn from these examples:\n"
        f"{examples_text}\n\n"
        f"Now classify:\n"
        f"Text: {text}\n"
        f"Sentiment:"
    )

    return ask_ai(
        prompt=prompt,
        system="You are a sentiment analysis expert.",
        temperature=0.1
    )


def classification_zero_shot(text: str) -> str:
    """Topic classification with zero shot"""
    prompt = (
        f"Classify this text into one of these categories:\n"
        f"SPORTS, TECHNOLOGY, POLITICS, ENTERTAINMENT, SCIENCE\n\n"
        f"Text: {text}\n"
        f"Category:"
    )

    return ask_ai(
        prompt=prompt,
        system="You are a text classification expert.",
        temperature=0.1
    )


def classification_few_shot(text: str) -> str:
    """Topic classification with few shot"""
    prompt = (
        f"Classify text into: SPORTS, TECHNOLOGY, "
        f"POLITICS, ENTERTAINMENT, SCIENCE\n\n"
        f"Examples:\n"
        f"Text: 'Apple releases new iPhone' → TECHNOLOGY\n"
        f"Text: 'Team wins championship' → SPORTS\n"
        f"Text: 'President signs new law' → POLITICS\n"
        f"Text: 'New Marvel movie released' → ENTERTAINMENT\n"
        f"Text: 'Scientists discover new planet' → SCIENCE\n\n"
        f"Now classify:\n"
        f"Text: {text}\n"
        f"Category:"
    )

    return ask_ai(
        prompt=prompt,
        system="You are a text classification expert.",
        temperature=0.1
    )


def format_zero_shot(raw_data: str) -> str:
    """Format data with zero shot"""
    prompt = (
        f"Convert this raw data into a "
        f"clean formatted JSON.\n\n"
        f"Raw data: {raw_data}\n"
        f"JSON:"
    )

    return ask_ai(
        prompt=prompt,
        system="You are a data formatting expert.",
        temperature=0.1
    )


def format_few_shot(raw_data: str) -> str:
    """Format data with few shot"""
    prompt = (
        f"Convert raw data into clean formatted JSON.\n\n"
        f"Example 1:\n"
        f"Raw: John Smith age 30 engineer\n"
        f"JSON: {{\"name\": \"John Smith\", "
        f"\"age\": 30, \"job\": \"engineer\"}}\n\n"
        f"Example 2:\n"
        f"Raw: Sarah Jones age 25 designer\n"
        f"JSON: {{\"name\": \"Sarah Jones\", "
        f"\"age\": 25, \"job\": \"designer\"}}\n\n"
        f"Now convert:\n"
        f"Raw: {raw_data}\n"
        f"JSON:"
    )

    return ask_ai(
        prompt=prompt,
        system="You are a data formatting expert.",
        temperature=0.1
    )


def run():
    """Zero Shot & Few Shot Feature"""

    # ✅ Header
    print_feature_header("Zero Shot & Few Shot")

    # ✅ Explain concept
    print_concept(
        "What is Zero/Few Shot?",
        "These are prompting techniques that control\n"
        "how many EXAMPLES you give to AI.\n\n"
        "Zero Shot:\n"
        "  • No examples given\n"
        "  • AI uses only training knowledge\n"
        "  • Fast but less accurate\n\n"
        "One Shot:\n"
        "  • ONE example given\n"
        "  • AI learns the pattern\n"
        "  • Better than zero shot\n\n"
        "Few Shot:\n"
        "  • 2-5 examples given\n"
        "  • AI learns pattern well\n"
        "  • Most accurate results\n\n"
        "Rule: More examples = Better accuracy!"
    )

    print_divider()

    # ✅ Demo 1: Sentiment Analysis
    print_step(
        "Demo 1",
        "Sentiment Analysis — Zero vs Few Shot"
    )

    print_info(
        "We will classify the SAME text\n"
        "using Zero Shot and Few Shot\n"
        "and compare results!"
    )

    test_texts = [
        "This movie was absolutely mind blowing!",
        "I waited 2 hours and the food was cold.",
        "The package arrived on time I suppose."
    ]

    for text in test_texts:
        print_divider()
        print_prompt(f"Text to classify: {text}")

        # Zero Shot
        print_step("Zero Shot", "No examples given...")
        print_thinking()

        zero_result = sentiment_zero_shot(text)
        print_response(
            f"[Zero Shot Result]\n\n"
            f"Text: {text}\n"
            f"Sentiment: {zero_result}"
        )

        # Few Shot
        print_step(
            "Few Shot",
            f"Using {len(SENTIMENT_EXAMPLES)} examples..."
        )

        print_info(
            "Examples being provided:\n" +
            "\n".join([
                f"  '{ex['text']}' → {ex['label']}"
                for ex in SENTIMENT_EXAMPLES
            ])
        )

        print_thinking()

        few_result = sentiment_few_shot(
            text=text,
            examples=SENTIMENT_EXAMPLES
        )

        print_response(
            f"[Few Shot Result]\n\n"
            f"Text: {text}\n"
            f"Sentiment: {few_result}"
        )

    print_divider()

    # ✅ Demo 2: Topic Classification
    print_step(
        "Demo 2",
        "Topic Classification — Zero vs Few Shot"
    )

    classify_texts = [
        "Scientists discover new treatment for cancer",
        "Local team wins the national championship",
        "New AI chip runs 10x faster than previous"
    ]

    for text in classify_texts:
        print_divider()
        print_prompt(f"Text: {text}")

        print_step("Zero Shot", "Classifying without examples...")
        print_thinking()

        zero_class = classification_zero_shot(text)

        print_response(
            f"[Zero Shot]\n\n"
            f"Category: {zero_class}"
        )

        print_step("Few Shot", "Classifying with examples...")
        print_thinking()

        few_class = classification_few_shot(text)

        print_response(
            f"[Few Shot]\n\n"
            f"Category: {few_class}"
        )

    print_divider()

    # ✅ Demo 3: Data Formatting
    print_step(
        "Demo 3",
        "Data Formatting — Zero vs Few Shot"
    )

    raw_inputs = [
        "Mike Johnson 28 years old software developer",
        "Emma Wilson 35 product manager at TechCorp"
    ]

    for raw in raw_inputs:
        print_divider()
        print_prompt(f"Raw Data: {raw}")

        print_step("Zero Shot", "Formatting without examples...")
        print_thinking()

        zero_format = format_zero_shot(raw)

        print_response(
            f"[Zero Shot Format]\n\n{zero_format}"
        )

        print_step("Few Shot", "Formatting with examples...")
        print_thinking()

        few_format = format_few_shot(raw)

        print_response(
            f"[Few Shot Format]\n\n{few_format}"
        )

    print_divider()

    # ✅ Demo 4: Custom test
    print_step(
        "Demo 4",
        "Test Zero vs Few Shot yourself!"
    )

    print_table(
        title="Choose a Task",
        columns=["Option", "Task"],
        rows=[
            ["1", "Sentiment Analysis"],
            ["2", "Topic Classification"],
            ["3", "Data Formatting"],
        ]
    )

    task_choice = get_user_input(
        "Choose task (1-3): "
    )

    custom_input = get_user_input(
        "📝 Enter your text/data: "
    )

    if not custom_input:
        custom_input = "The new update completely broke my workflow"
        print_info(f"Using default: {custom_input}")

    print_divider()

    if task_choice == "1":
        print_step("Zero Shot", "Sentiment...")
        print_thinking()
        z = sentiment_zero_shot(custom_input)
        print_response(f"[Zero Shot]\nSentiment: {z}")

        print_step("Few Shot", "Sentiment with examples...")
        print_thinking()
        f = sentiment_few_shot(custom_input, SENTIMENT_EXAMPLES)
        print_response(f"[Few Shot]\nSentiment: {f}")

    elif task_choice == "2":
        print_step("Zero Shot", "Classifying...")
        print_thinking()
        z = classification_zero_shot(custom_input)
        print_response(f"[Zero Shot]\nCategory: {z}")

        print_step("Few Shot", "Classifying with examples...")
        print_thinking()
        f = classification_few_shot(custom_input)
        print_response(f"[Few Shot]\nCategory: {f}")

    elif task_choice == "3":
        print_step("Zero Shot", "Formatting...")
        print_thinking()
        z = format_zero_shot(custom_input)
        print_response(f"[Zero Shot]\nFormatted: {z}")

        print_step("Few Shot", "Formatting with examples...")
        print_thinking()
        f = format_few_shot(custom_input)
        print_response(f"[Few Shot]\nFormatted: {f}")

    else:
        print_step("Zero Shot", "Sentiment...")
        print_thinking()
        z = sentiment_zero_shot(custom_input)
        print_response(f"[Zero Shot]\nSentiment: {z}")

        print_step("Few Shot", "Sentiment with examples...")
        print_thinking()
        f = sentiment_few_shot(custom_input, SENTIMENT_EXAMPLES)
        print_response(f"[Few Shot]\nSentiment: {f}")

    print_divider()

    # ✅ Summary
    print_concept(
        "Zero Shot & Few Shot Summary",
        "When to use what:\n\n"
        "Zero Shot:\n"
        "  ✅ Simple well known tasks\n"
        "  ✅ When speed matters\n"
        "  ✅ General questions\n"
        "  ❌ Complex custom tasks\n\n"
        "One Shot:\n"
        "  ✅ Show AI the pattern once\n"
        "  ✅ Custom output format\n"
        "  ✅ Quick improvement\n\n"
        "Few Shot:\n"
        "  ✅ Complex classification\n"
        "  ✅ Custom formatting\n"
        "  ✅ Domain specific tasks\n"
        "  ✅ When accuracy is critical\n\n"
        "Remember:\n"
        "  More examples = Better results\n"
        "  But more tokens used too!"
    )