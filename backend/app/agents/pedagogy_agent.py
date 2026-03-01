import re
from app.services.llm_service import generate_response

LANG_NAMES = {"c": "C", "cpp": "C++", "python": "Python", "java": "Java", "csharp": "C#"}


def pedagogy_agent(code: str, mode: str, language: str = "c") -> str:
    lang_name = LANG_NAMES.get(language, language.upper())

    if mode == "interview":
        instruction = f"""Explain the {lang_name} code at interview level.

STRICT FORMAT:
- Exactly 5 bullet points.
- Each bullet must start with "- ".
- Each bullet under 25 words.
- Total under 120 words.
- Cover:
  1. Overall logic
  2. Key algorithm idea
  3. Time complexity
  4. Space complexity
  5. Important edge case
- No repetition.
"""
        model = "llama-3.3-70b-versatile"
        max_tokens = 220
    else:
        instruction = f"""Explain the {lang_name} code for a beginner.

STRICT FORMAT:
- 4 to 6 bullet points.
- Each bullet must start with "- ".
- Each bullet under 20 words.
- Total under 100 words.
- Keep language simple.
- No repetition.
"""
        model = "llama-3.1-8b-instant"
        max_tokens = 180

    prompt = f"""You are a disciplined programming mentor.

Follow ALL rules strictly.
Do not output paragraphs.
Do not use numbering.
Only use bullet points.

{instruction}

Code:
{code}
"""

    response = generate_response(prompt, model, max_tokens)
    response = response.strip()
    response = re.sub(r"^```.*?\n", "", response)
    response = re.sub(r"```$", "", response)

    lines = response.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not line.startswith("-"):
            line = "- " + line.lstrip("- ").strip()
        cleaned.append(line)

    return "\n".join(cleaned)
