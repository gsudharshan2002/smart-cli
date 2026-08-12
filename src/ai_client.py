from openai import OpenAI
from src.config import (
    AI_API_KEY,
    AI_MODEL,
    BASE_URL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_P
)

client = OpenAI(
    api_key=AI_API_KEY,
    base_url=BASE_URL
)



def ask_ai(
    prompt: str,
    system: str = "You are a helpful assistant.",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    top_p: float = DEFAULT_TOP_P,
    model: str = AI_MODEL
) -> str:
    """
     Core function to call AI
    - prompt     → user message
    - system     → system role message
    - temperature → creativity level
    - max_tokens  → response length
    - top_p       → nucleus sampling
    - model       → AI model to use
    """


    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )

    return response.choices[0].message.content

def ask_ai_with_history(
    messages: list,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    model: str = AI_MODEL
) -> str:
    """
    ✅ Call AI with full message history
    Used for multi-turn conversations
    """

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content