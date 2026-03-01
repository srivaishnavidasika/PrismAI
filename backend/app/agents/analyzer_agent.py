import json
import re
from app.services.llm_service import generate_response

LANG_NAMES = {"c": "C", "cpp": "C++", "python": "Python", "java": "Java", "csharp": "C#"}

def _extract_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def analyzer_agent(code: str, language: str = "c") -> dict:
    lang_name = LANG_NAMES.get(language, language.upper())

    prompt = f"""You are a strict {lang_name} code analysis expert.

Analyze the given {lang_name} code.

Return STRICT JSON ONLY in this format:

{{
    "syntax_errors": ["error1", "error2"],
    "logical_errors": ["error1", "error2"],
    "inefficiencies": ["issue1", "issue2"],
    "summary": "short overall quality summary"
}}

Rules:
- Analyze the code ONLY based on its own logic, syntax, and structure.
- Do NOT assume what the code is supposed to output — judge it purely as written.
- Do NOT flag missing JSON output, return formats, or expected outputs unless explicitly required by the code itself.
- If no syntax errors, return empty list.
- If no logical errors, return empty list.
- If no inefficiencies, return empty list.
- Do NOT output markdown.
- Do NOT output text outside JSON.

Code:
{code}
"""

    response = generate_response(prompt, "llama-3.1-8b-instant", 400)
    cleaned = _extract_json(response)

    try:
        parsed = json.loads(cleaned)
        return {
            "syntax_errors":  parsed.get("syntax_errors", []),
            "logical_errors": parsed.get("logical_errors", []),
            "inefficiencies": parsed.get("inefficiencies", []),
            "summary":        parsed.get("summary", "Analysis completed.")
        }
    except Exception:
        return {
            "syntax_errors": [], "logical_errors": [], "inefficiencies": [],
                    "summary": "Analysis failed. Raw output:\n" + response[:500]
        }