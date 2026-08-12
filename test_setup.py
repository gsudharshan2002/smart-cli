# test_setup.py

from src.config import APP_NAME, AI_MODEL
from src.ai_client import ask_ai
from src.utils.printer import (
    print_welcome,
    print_prompt,
    print_response,
    print_thinking
)

# Test printer
print_welcome(APP_NAME)

# Test AI
prompt = "Say hello in one sentence!"
print_prompt(prompt)
print_thinking()

response = ask_ai(prompt)
print_response(response)