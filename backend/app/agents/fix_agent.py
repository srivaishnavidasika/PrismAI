import json
import re
from app.services.llm_service import generate_response

LANG_NAMES = {"c": "C", "cpp": "C++", "python": "Python", "java": "Java", "csharp": "C#"}


def fix_agent(code: str, language: str = "c") -> str:
    lang_name = LANG_NAMES.get(language, language.upper())

    prompt = f"""You are a strict {lang_name} debugging expert.

Analyze the given {lang_name} code carefully.

IMPORTANT RULES:
- Only report REAL syntax errors, logical bugs, or runtime issues.
- Do NOT invent problems.
- Do NOT suggest stylistic improvements.
- If the code is already correct, say so.

Return STRICT JSON ONLY in one of the following formats:

IF A REAL BUG EXISTS:

{{
    "issue": "brief description of the real bug",
    "corrected_code": "FULL corrected code here",
    "explanation": "clear explanation of the fix"
}}

IF NO BUG EXISTS:

{{
    "issue": null,
    "corrected_code": null,
    "explanation": "The code is already correct and does not require any fixes."
}}

CRITICAL corrected_code rules:
- Use \\n ONLY between lines of code — NEVER inside a string literal.
- WRONG: printf("Prime\\n"); — this splits the string and breaks the code.
- CORRECT: printf("Prime\\n"); must appear on ONE line with the \\n INSIDE the quotes on that same line.
- Actually: keep ALL content inside quotes on a single unbroken line. No literal newline may appear inside a quoted string.
- Do NOT output markdown. Do NOT output text outside JSON.

Code:
{code}
"""

    response = generate_response(prompt, "llama-3.1-8b-instant", 440)

    response = response.strip()
    response = re.sub(r"^```json\s*", "", response, flags=re.MULTILINE)
    response = re.sub(r"^```\s*", "", response, flags=re.MULTILINE)
    response = re.sub(r"```\s*$", "", response, flags=re.MULTILINE)
    response = response.strip()

    # Extract JSON if model added surrounding text
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        response = match.group(0)

    try:
        parsed = json.loads(response)
        issue = parsed.get("issue")
        corrected_code = parsed.get("corrected_code")
        explanation = parsed.get("explanation")

        if issue in ["No issues found.", "No issues found", None]:
            return json.dumps({
                "issue": None, "corrected_code": None,
                "explanation": explanation or "The code is already correct and does not require any fixes."
            })

        # Decode escaped newlines so frontend renders real code, not \n literals
        if corrected_code and isinstance(corrected_code, str):
            corrected_code = corrected_code.replace("\\n", "\n").replace("\\t", "\t")
            # Fix split string literals: if a line ends mid-string (odd number of quotes),
            # the LLM incorrectly inserted a newline inside the string. Rejoin those lines.
            import re as _re
            fixed_lines = []
            pending = None
            for line in corrected_code.split("\n"):
                if pending is not None:
                    pending += line
                    if pending.count('"') % 2 == 0:
                        fixed_lines.append(pending)
                        pending = None
                else:
                    if line.count('"') % 2 != 0:
                        pending = line
                    else:
                        fixed_lines.append(line)
            if pending is not None:
                fixed_lines.append(pending)
            corrected_code = "\n".join(fixed_lines)
        return json.dumps({"issue": issue, "corrected_code": corrected_code, "explanation": explanation})

    except json.JSONDecodeError:
        issue_match = re.search(r'"issue"\s*:\s*"([^"]+)"', response)
        explanation_match = re.search(r'"explanation"\s*:\s*"([^"]+)"', response)
        corrected_match = re.search(r'"corrected_code"\s*:\s*(.*?)(?=\s*"explanation")', response, re.DOTALL)

        corrected_code = None
        if corrected_match:
            corrected_code = corrected_match.group(1).strip().strip(",").strip('"')

        return json.dumps({
            "issue": issue_match.group(1) if issue_match else None,
            "corrected_code": corrected_code,
            "explanation": explanation_match.group(1) if explanation_match else "Unable to parse explanation."
        })