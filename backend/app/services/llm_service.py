import os
from groq import Groq
from dotenv import load_dotenv

# --------------------------------------------------
# Environment Setup
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

client = Groq(api_key=api_key)

DEFAULT_TIMEOUT = 25  # seconds per individual LLM call


# --------------------------------------------------
# Content Extractor
# --------------------------------------------------

def _extract_content(choice) -> str:
    raw_content = choice.message.content

    if isinstance(raw_content, str):
        return raw_content.strip()

    if isinstance(raw_content, list):
        text_parts = []
        for item in raw_content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
        return "\n".join(text_parts).strip()

    return ""


# --------------------------------------------------
# Internal LLM Call
# --------------------------------------------------

def _call_llm(prompt: str, model: str, max_tokens: int) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a disciplined programming mentor. "
                        "Follow instructions strictly. "
                        "Return only what is requested. "
                        "Do not add extra commentary."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0,
            top_p=1,
            timeout=DEFAULT_TIMEOUT,
        )

        if not response.choices:
            return "Error: No response from LLM."

        choice = response.choices[0]
        content = _extract_content(choice)

        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            content += "\n\n[Output truncated due to token limit.]"

        return content or "Error: Empty LLM response."

    except Exception:
        # Do NOT leak internal errors to frontend
        return "Error: LLM service unavailable."


# --------------------------------------------------
# Public API
# --------------------------------------------------

def generate_response(prompt: str, model: str, max_tokens: int) -> str:
    """
    Fully synchronous, deadlock-proof LLM call.
    No threadpool.
    No nested executors.
    FastAPI handles concurrency safely.
    """
    return _call_llm(prompt, model, max_tokens)