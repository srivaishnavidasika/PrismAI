import json
import re
from app.services.llm_service import generate_response
from app.memory.memory_store import get_all_language_memory

LANG_NAMES = {"c": "C", "cpp": "C++", "python": "Python", "java": "Java", "csharp": "C#"}


def mistake_fixer_agent(user_id: str = "default_user", language: str = "c") -> dict:
    lang_name = LANG_NAMES.get(language, language.upper())

    full_memory = get_all_language_memory(user_id) or {}
    lang_memory = full_memory.get("by_language", {}).get(language, {})
    common_mistakes = lang_memory.get("common_mistakes", [])

    if not common_mistakes:
        return {
            "no_data": True,
            "questions": [],
            "mistakes_targeted": [],
            "language": lang_name
        }

    mistakes_str = "\n".join(f"- {m}" for m in common_mistakes)

    prompt = f"""You are a strict {lang_name} coding mentor reviewing a student's mistake history.

The following mistakes have been observed repeatedly in this student's past {lang_name} coding sessions specifically:
{mistakes_str}

Generate exactly 5 targeted diagnostic exercises that directly address these recurring {lang_name} mistakes.

Each exercise must:
- Be specifically designed to expose and fix one of the listed mistakes
- Ask the student to identify, explain, or correct a broken/flawed {lang_name} code snippet
- Focus on understanding WHY the mistake happens and how to avoid it

Return STRICT JSON ONLY:

{{
    "questions": [
        {{
            "question": "Full exercise text — include a short broken {lang_name} code snippet if relevant",
            "targets": "Exact mistake from the list this exercise addresses"
        }}
    ]
}}

Rules:
- Exactly 5 questions.
- Do NOT output markdown.
- Do NOT output text outside JSON.
"""

    response = generate_response(prompt, "llama-3.1-8b-instant", 700)
    response = response.strip()
    response = re.sub(r"^```json\s*", "", response, flags=re.MULTILINE)
    response = re.sub(r"^```\s*", "", response, flags=re.MULTILINE)
    response = re.sub(r"```\s*$", "", response, flags=re.MULTILINE)
    response = response.strip()

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        response = match.group(0)

    try:
        data = json.loads(response)
        questions = data.get("questions", [])
        if not isinstance(questions, list):
            questions = []

        cleaned = []
        for q in questions[:5]:
            if isinstance(q, dict) and "question" in q:
                cleaned.append({
                    "question": str(q.get("question", "")),
                    "targets":  str(q.get("targets", ""))
                })

        return {
            "no_data":          False,
            "questions":        cleaned,
            "mistakes_targeted": common_mistakes,
            "language":         lang_name
        }

    except json.JSONDecodeError:
        return {
            "no_data":          False,
            "questions":        [],
            "mistakes_targeted": common_mistakes,
            "language":         lang_name,
            "error":            "Failed to generate questions. Please try again."
        }
