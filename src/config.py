import os
from dotenv import load_dotenv

load_dotenv()

APP_NAME = os.getenv("APP_NAME", "Smart CLI")
AI_API_KEY = os.getenv("AI_API_KEY")
AI_MODEL = os.getenv("AI_MODEL", "llama3-8b-8192")

BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
RAG_MAX_TOKENS = 4096
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 40

if not AI_API_KEY:
    raise ValueError("❌ AI_API_KEY not found in .env file!")