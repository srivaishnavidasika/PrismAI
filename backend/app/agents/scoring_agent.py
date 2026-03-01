import json
import re
from app.services.llm_service import generate_response

LANG_NAMES = {"c": "C", "cpp": "C++", "python": "Python", "java": "Java", "csharp": "C#"}


def scoring_agent(code: str, language: str = "c", analysis: dict | None = None) -> dict:
    lang_name = LANG_NAMES.get(language, language.upper())

    analysis_context = ""
    if analysis:
        errors = []
        if analysis.get("syntax_errors"):
            errors.append(f"Syntax errors: {'; '.join(analysis['syntax_errors'])}")
        if analysis.get("logical_errors"):
            errors.append(f"Logical errors: {'; '.join(analysis['logical_errors'])}")
        if analysis.get("inefficiencies"):
            errors.append(f"Inefficiencies: {'; '.join(analysis['inefficiencies'])}")
        if analysis.get("summary"):
            errors.append(f"Summary: {analysis['summary']}")
        if errors:
            analysis_context = "\n\nPre-analysis findings:\n" + "\n".join(errors)

    prompt = f"""You are a strict {lang_name} code evaluator.

Evaluate the given {lang_name} code and return STRICT JSON in this format:

{{
    "syntax_score": 0-10,
    "logic_score": 0-10,
    "clarity_score": 0-10,
    "robustness_score": 0-10,
    "overall_score": 0-10
}}

Rules:
- Scores must be integers except overall_score which may be decimal.
- Be realistic and consistent.
- Do not include explanations.
- Return only valid JSON.

Code:
{code}{analysis_context}
"""

    response = generate_response(prompt, "llama-3.1-8b-instant", 210)

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

        def clamp(v):
            return max(0, min(10, v)) if isinstance(v, (int, float)) else 0

        return {
            "syntax_score":     clamp(data.get("syntax_score", 0)),
            "logic_score":      clamp(data.get("logic_score", 0)),
            "clarity_score":    clamp(data.get("clarity_score", 0)),
            "robustness_score": clamp(data.get("robustness_score", 0)),
            "overall_score":    clamp(data.get("overall_score", 0)),
        }
    except json.JSONDecodeError:
        return {"error": "Invalid JSON returned by scoring agent.", "raw_response": response}