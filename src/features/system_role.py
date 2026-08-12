# src/features/system_role.py

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


# ✅ Predefined System Roles
SYSTEM_ROLES = {
    "1": {
        "name": "Expert Doctor 🏥",
        "role": """You are Dr. Smith, an experienced medical doctor 
with 20 years of practice. You explain medical concepts 
clearly to patients. You always:
- Use simple language
- Show empathy
- Recommend consulting a real doctor
- Never diagnose but educate
- Keep responses professional and caring"""
    },

    "2": {
        "name": "Strict Teacher 📚",
        "role": """You are Professor Johnson, a very strict but 
fair teacher with high standards. You always:
- Correct mistakes immediately
- Demand proper grammar
- Give detailed explanations
- Ask follow up questions
- Encourage critical thinking
- Never accept lazy answers"""
    },

    "3": {
        "name": "Funny Comedian 😂",
        "role": """You are Mike the comedian. You answer every 
question with humor and jokes. You always:
- Make jokes about the topic
- Use funny analogies
- Add punchlines
- Keep it light and entertaining
- Still give correct info but in funny way
- Use emojis and humor"""
    },

    "4": {
        "name": "Professional Chef 👨‍🍳",
        "role": """You are Chef Antonio, a Michelin star chef 
with 30 years experience in Italian cuisine. You always:
- Talk passionately about food
- Give detailed cooking tips
- Mention ingredients and techniques
- Share cooking secrets
- Use cooking terminology
- Express love for food"""
    },

    "5": {
        "name": "Startup CEO 💼",
        "role": """You are Alex, a successful startup CEO who 
has built 3 unicorn companies. You always:
- Think in terms of growth and scale
- Use startup terminology
- Talk about disruption and innovation
- Give bold business advice
- Mention metrics and KPIs
- Stay optimistic and ambitious"""
    },

    "6": {
        "name": "Pirate Captain 🏴‍☠️",
        "role": """You are Captain Blackbeard, a legendary pirate. 
You answer every question as a pirate. You always:
- Use pirate language (Arr! Ahoy! Matey!)
- Replace words with pirate versions
- Relate everything to the sea
- Talk about treasure and adventure
- Stay in character completely
- Make it fun and dramatic"""
    },

    "7": {
        "name": "Custom Role ✏️",
        "role": None  # User defines this
    }
}


def demo_same_question_different_roles(question: str):
    """
    Ask same question to different AI personalities
    Show how system role changes everything
    """

    roles_to_demo = ["1", "3", "6"]  # Doctor, Comedian, Pirate

    for role_key in roles_to_demo:
        role_info = SYSTEM_ROLES[role_key]

        print_step(
            f"Role: {role_info['name']}",
            "Asking same question..."
        )

        print_prompt(
            f"Question: {question}\n"
            f"System Role: {role_info['name']}"
        )

        print_thinking()

        response = ask_ai(
            prompt=question,
            system=role_info["role"]
        )

        print_response(
            f"[{role_info['name']}]\n\n{response}"
        )

        print_divider()


def interactive_role_chat(
    role_name: str,
    system_role: str,
    rounds: int = 3
):
    """
    Have a multi turn conversation
    with a specific AI role
    """

    print_info(
        f"Starting chat with {role_name}\n"
        f"You will have {rounds} conversation rounds!"
    )

    print_divider()

    # ✅ Build conversation history
    messages = [
        {"role": "system", "content": system_role}
    ]

    for round_num in range(1, rounds + 1):
        print_step(
            f"Round {round_num}/{rounds}",
            f"Chat with {role_name}"
        )

        user_input = get_user_input(
            f"💬 You: "
        )

        if not user_input:
            user_input = "Tell me something interesting!"
            print_info(f"Using default: {user_input}")

        # ✅ Add user message
        messages.append({
            "role": "user",
            "content": user_input
        })

        print_thinking()

        # ✅ Get AI response with full history
        from src.ai_client import ask_ai_with_history
        response = ask_ai_with_history(
            messages=messages,
            temperature=0.8
        )

        # ✅ Add AI response to history
        messages.append({
            "role": "assistant",
            "content": response
        })

        print_response(
            f"[{role_name}]\n\n{response}"
        )

    return messages


def run():
    """System Role Logic Feature"""

    # ✅ Header
    print_feature_header("System Role Logic")

    # ✅ Explain concept
    print_concept(
        "What is System Role?",
        "System Role = Identity & personality you give AI\n"
        "BEFORE the conversation starts.\n\n"
        "It controls:\n"
        "  🎭 Personality → Who AI pretends to be\n"
        "  🧠 Expertise  → What AI specializes in\n"
        "  🗣️  Tone       → How AI talks\n"
        "  📏 Rules      → What AI must/cannot do\n"
        "  🎯 Focus      → What AI cares about\n\n"
        "Same question + Different role = Completely different answer!\n\n"
        "This is the MOST powerful prompt technique!"
    )

    print_divider()

    # ✅ Demo 1: Same question different roles
    print_step(
        "Demo 1",
        "Same question → 3 different AI personalities"
    )

    same_question = "How should I deal with stress?"

    print_info(
        f"Question: '{same_question}'\n"
        f"We will ask 3 different AI roles!\n"
        f"  1. Expert Doctor 🏥\n"
        f"  2. Funny Comedian 😂\n"
        f"  3. Pirate Captain 🏴‍☠️"
    )

    print_divider()

    demo_same_question_different_roles(same_question)

    print_success(
        "See how SAME question gets completely "
        "different answers based on system role!"
    )

    print_divider()

    # ✅ Demo 2: Show all available roles
    print_step(
        "Demo 2",
        "Available Roles — Pick one to chat with!"
    )

    print_table(
        title="🎭 Available AI Roles",
        columns=["Option", "Role", "Style"],
        rows=[
            ["1", "Expert Doctor 🏥", "Professional & caring"],
            ["2", "Strict Teacher 📚", "Demanding & detailed"],
            ["3", "Funny Comedian 😂", "Humorous & entertaining"],
            ["4", "Professional Chef 👨‍🍳", "Passionate & technical"],
            ["5", "Startup CEO 💼", "Ambitious & strategic"],
            ["6", "Pirate Captain 🏴‍☠️", "Dramatic & fun"],
            ["7", "Custom Role ✏️", "You define it!"],
        ]
    )

    role_choice = get_user_input(
        "🎭 Choose a role (1-7): "
    )

    if role_choice not in SYSTEM_ROLES:
        role_choice = "3"
        print_info("Invalid choice. Using Comedian!")

    selected_role = SYSTEM_ROLES[role_choice]

    # ✅ Handle custom role
    if role_choice == "7":
        print_step(
            "Custom Role",
            "Define your own AI personality!"
        )

        role_name = get_user_input(
            "📝 Role name (e.g. 'Space Explorer'): "
        )
        if not role_name:
            role_name = "Space Explorer 🚀"

        role_description = get_user_input(
            "📝 Describe the role behavior: "
        )
        if not role_description:
            role_description = (
                "You are a space explorer who has "
                "visited every planet. You relate "
                "everything to space and adventure."
            )

        selected_role = {
            "name": role_name,
            "role": role_description
        }

        print_success(f"Custom role created: {role_name}")

    print_divider()

    # ✅ Demo 3: Quick single question test
    print_step(
        "Demo 3",
        f"Test {selected_role['name']} with one question"
    )

    test_question = get_user_input(
        f"📝 Ask {selected_role['name']} anything: "
    )

    if not test_question:
        test_question = "What is the meaning of life?"
        print_info(f"Using default: {test_question}")

    print_prompt(
        f"Role: {selected_role['name']}\n"
        f"Question: {test_question}"
    )

    print_thinking()

    single_response = ask_ai(
        prompt=test_question,
        system=selected_role["role"],
        temperature=0.8
    )

    print_response(
        f"[{selected_role['name']}]\n\n"
        f"{single_response}"
    )

    print_divider()

    # ✅ Demo 4: Multi turn chat
    print_step(
        "Demo 4",
        f"Multi-turn chat with {selected_role['name']}"
    )

    print_info(
        "Now have a REAL conversation!\n"
        "AI remembers previous messages\n"
        "in this chat session."
    )

    want_chat = get_user_input(
        f"💬 Start multi-turn chat with "
        f"{selected_role['name']}? (yes/no): "
    )

    if want_chat.lower() in ["yes", "y", ""]:
        chat_history = interactive_role_chat(
            role_name=selected_role["name"],
            system_role=selected_role["role"],
            rounds=3
        )

        print_success(
            f"Chat complete! "
            f"Total messages: {len(chat_history)}"
        )

    print_divider()

    # ✅ Demo 5: Role comparison table
    print_step(
        "Demo 5",
        "How roles affect AI behavior"
    )

    compare_question = "Explain what is Python?"

    print_info(
        f"Question: '{compare_question}'\n"
        f"Asking Doctor and Comedian..."
    )

    roles_compare = [
        SYSTEM_ROLES["1"],  # Doctor
        SYSTEM_ROLES["3"]   # Comedian
    ]

    for role in roles_compare:
        print_step(role["name"], "Responding...")
        print_thinking()

        compare_response = ask_ai(
            prompt=compare_question,
            system=role["role"],
            temperature=0.7
        )

        print_response(
            f"[{role['name']}]\n\n{compare_response}"
        )

    print_divider()

    # ✅ Summary
    print_concept(
        "System Role Summary",
        "Key Takeaways:\n\n"
        "System Role Controls:\n"
        "  🎭 Who AI is\n"
        "  🗣️  How AI talks\n"
        "  🧠 What AI knows\n"
        "  📏 How AI behaves\n\n"
        "Best Practices:\n"
        "  ✅ Be specific about personality\n"
        "  ✅ Define clear rules\n"
        "  ✅ Set tone & style\n"
        "  ✅ Give background context\n"
        "  ✅ Define what AI must NOT do\n\n"
        "Real World Uses:\n"
        "  ✅ Customer support bots\n"
        "  ✅ Educational assistants\n"
        "  ✅ Entertainment chatbots\n"
        "  ✅ Domain expert systems\n"
        "  ✅ Brand voice consistency"
    )