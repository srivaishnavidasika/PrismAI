import json
import re
from app.services.llm_service import generate_response
from app.memory.memory_store import get_user_memory

LANG_NAMES = {"c": "C", "cpp": "C++", "python": "Python", "java": "Java", "csharp": "C#"}


def practice_agent(code: str, mode: str, user_id: str = "default_user", language: str = "c") -> dict:
    lang_name = LANG_NAMES.get(language, language.upper())

    # Now reads language-specific memory
    memory = get_user_memory(user_id, language) or {}
    common_mistakes = memory.get("common_mistakes", [])
    recent_scores   = memory.get("recent_scores", [])

    valid_scores = [s for s in recent_scores if isinstance(s, (int, float))]
    struggling   = bool(valid_scores) and (sum(valid_scores) / len(valid_scores)) < 6

    difficulty_instruction = (
        f"Keep problems simple and focused on core {lang_name} concepts."
        if mode == "beginner"
        else f"Include deeper reasoning and possible edge cases for {lang_name}."
    )
    if struggling:
        difficulty_instruction += " Do NOT increase difficulty. Keep explanations clearer."

    memory_focus = ""
    if common_mistakes:
        memory_focus = f"Focus on reinforcing these {lang_name}-specific mistakes: {', '.join(common_mistakes)}."

    prompt = f"""You are an adaptive {lang_name} programming mentor.

Generate practice problems for the {lang_name} code below.

Return ONLY this JSON, no other text:
{{
    "similar_problems": ["problem1", "problem2"],
    "challenge_problem": "challenge text",
    "hints": ["hint1", "hint2", "hint3"]
}}

Rules:
- Return valid JSON only. No markdown. No explanation. No extra keys.
- similar_problems: exactly 2 short problem descriptions.
- challenge_problem: one harder problem description.
- hints: exactly 3 short hints.

{difficulty_instruction}
{memory_focus}

Code:
{code}
"""

    model      = "llama-3.3-70b-versatile" if mode == "interview" else "llama-3.1-8b-instant"
    max_tokens = 500

    response = generate_response(prompt, model, max_tokens)
    print(f"[PRACTICE AGENT] lang={language} mode={mode} raw: {repr(response[:200])}")

    response = response.strip()
    response = re.sub(r"^```json\s*", "", response, flags=re.MULTILINE)
    response = re.sub(r"^```\s*",    "", response, flags=re.MULTILINE)
    response = re.sub(r"```\s*$",    "", response, flags=re.MULTILINE)
    response = response.strip()

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        response = match.group(0)

    try:
        parsed = json.loads(response)
        return {
            "similar_problems":  parsed.get("similar_problems", []),
            "challenge_problem": parsed.get("challenge_problem", ""),
            "hints":             parsed.get("hints", [])
        }
    except json.JSONDecodeError as e:
        print(f"[PRACTICE AGENT] parse FAILED: {e}")
        return {
            "similar_problems": [], "challenge_problem": "", "hints": [],
            "error": f"Practice generation failed: {str(e)}"
        }
